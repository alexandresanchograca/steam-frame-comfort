use super::*;
use bytemuck::bytes_of;
use comfort_protocol::{EyeUse, FRAME_SIZE_V5, Header, ImageRemove, MAX_IMAGES};
use rustix::{
    io::{FdFlags, fcntl_getfd},
    net::{SendAncillaryBuffer, SendAncillaryMessage, SendFlags, sendmsg, socketpair},
};
use std::{
    fs,
    io::IoSlice,
    os::{
        fd::{AsFd, AsRawFd, BorrowedFd},
        unix::net::UnixStream,
    },
    path::PathBuf,
};

fn connected_pair() -> (SourceClient, OwnedFd) {
    let (client, server) = socketpair(
        AddressFamily::UNIX,
        SocketType::SEQPACKET,
        SocketFlags::NONBLOCK | SocketFlags::CLOEXEC,
        None,
    )
    .unwrap();
    (
        SourceClient {
            socket: Some(client),
            ..Default::default()
        },
        server,
    )
}

fn send(server: &OwnedFd, bytes: &[u8], fds: &[BorrowedFd<'_>]) {
    let mut storage = [MaybeUninit::uninit(); rustix::cmsg_space!(ScmRights(8))];
    let mut control = SendAncillaryBuffer::new(&mut storage);
    if !fds.is_empty() {
        assert!(control.push(SendAncillaryMessage::ScmRights(fds)));
    }
    assert_eq!(
        sendmsg(
            server,
            &[IoSlice::new(bytes)],
            &mut control,
            SendFlags::NOSIGNAL
        )
        .unwrap(),
        bytes.len()
    );
}

fn identity(fd: &impl AsRawFd) -> PathBuf {
    fs::read_link(format!("/proc/self/fd/{}", fd.as_raw_fd())).unwrap()
}

fn descriptor_count(identity: &Path) -> usize {
    fs::read_dir("/proc/self/fd")
        .unwrap()
        .filter_map(Result::ok)
        .filter(|entry| fs::read_link(entry.path()).is_ok_and(|target| target == identity))
        .count()
}

fn add(id: u64) -> ImageAdd {
    ImageAdd {
        header: Header::observer::<ImageAdd>(2),
        id,
        width: 2304,
        height: 2304,
        ..ImageAdd::default()
    }
}

#[test]
fn source_updates_replace_and_release_received_descriptors() {
    let (mut client, server) = connected_pair();
    let (first, replacement) = UnixStream::pair().unwrap();
    let image = add(5);
    send(&server, bytes_of(&image), &[first.as_fd()]);
    client.poll(1).unwrap();
    let received = &client.images[&5];
    let revision = received.revision;
    let first_identity = identity(&first);
    assert_eq!(identity(&received.fd), first_identity);
    assert!(
        fcntl_getfd(&received.fd)
            .unwrap()
            .contains(FdFlags::CLOEXEC)
    );
    assert_eq!(descriptor_count(&first_identity), 2);

    let mut update = image;
    update.ycbcr.valid = 1;
    send(&server, bytes_of(&update), &[replacement.as_fd()]);
    client.poll(2).unwrap();
    let replacement_identity = identity(&replacement);
    let received = &client.images[&5];
    assert_eq!(received.description, update);
    assert!(received.revision > revision);
    assert_eq!(identity(&received.fd), replacement_identity);
    assert_eq!(descriptor_count(&first_identity), 1);

    send(&server, bytes_of(&add(8)), &[]);
    send(
        &server,
        bytes_of(&add(9)),
        &[first.as_fd(), replacement.as_fd()],
    );
    client.poll(3).unwrap();
    assert_eq!(client.protocol_errors, 2);
    assert_eq!(client.images.len(), 1);
    assert_eq!(descriptor_count(&first_identity), 1);
    assert_eq!(descriptor_count(&replacement_identity), 2);

    let remove = ImageRemove {
        header: Header::observer::<ImageRemove>(3),
        id: 5,
    };
    send(&server, bytes_of(&remove), &[]);
    client.poll(4).unwrap();
    assert!(client.images.is_empty());
    assert_eq!(client.take_removals(), [5]);
    assert!(client.take_removals().is_empty());
    assert_eq!(descriptor_count(&replacement_identity), 1);
}

#[test]
fn source_frames_preserve_labels_and_timestamp_only_valid_updates() {
    let (mut client, server) = connected_pair();
    let mut hello = Hello {
        header: Header::observer::<Hello>(1),
        pid: 42,
        ..Hello::default()
    };
    hello.header.version = 5;
    hello.device_uuid[3] = 7;
    send(&server, bytes_of(&hello), &[]);
    let mut frame = Frame {
        header: Header::observer::<Frame>(4),
        sequence: 17,
        touched_count: 1,
        ..Frame::default()
    };
    frame.touched_ids[0] = 5;
    frame.touched_eyes[0] = EyeUse {
        eyes: 1,
        slot: 9,
        layers: [1, 0],
    };
    send(&server, bytes_of(&frame), &[]);
    client.poll(150).unwrap();
    assert_eq!(client.hello, Some(hello));
    assert_eq!(client.frame, Some(frame));
    assert_eq!(client.frame_received_ns, 150);

    frame.header.version = 5;
    frame.header.size = FRAME_SIZE_V5 as u32;
    frame.sequence += 1;
    send(&server, &bytes_of(&frame)[..FRAME_SIZE_V5], &[]);
    client.poll(250).unwrap();
    frame.touched_eyes[0] = EyeUse::default();
    assert_eq!(client.frame, Some(frame));
    assert_eq!(client.frame_received_ns, 250);

    let mut invalid = frame;
    invalid.touched_count = MAX_IMAGES as u32 + 1;
    send(&server, &bytes_of(&invalid)[..FRAME_SIZE_V5], &[]);
    client.poll(350).unwrap();
    assert_eq!(client.protocol_errors, 1);
    assert_eq!(client.frame, Some(frame));
    assert_eq!(client.frame_received_ns, 250);
}

#[test]
fn invalid_datagrams_and_eof_disconnect_and_release_descriptors() {
    let (attachment, _peer) = UnixStream::pair().unwrap();
    let attachment_identity = identity(&attachment);
    for (invalid_bytes, ancillary_count, expected) in [
        (24, 1, io::ErrorKind::InvalidData),
        (MAX_PACKET_SIZE + 1, 1, io::ErrorKind::InvalidData),
        (size_of::<ImageAdd>(), 8, io::ErrorKind::InvalidData),
        (0, 0, io::ErrorKind::UnexpectedEof),
    ] {
        let (mut client, server) = connected_pair();
        let image = add(5);
        send(&server, bytes_of(&image), &[attachment.as_fd()]);
        let hello = Hello {
            header: Header::observer::<Hello>(1),
            ..Hello::default()
        };
        send(&server, bytes_of(&hello), &[]);
        let frame = Frame {
            header: Header::observer::<Frame>(4),
            ..Frame::default()
        };
        send(&server, bytes_of(&frame), &[]);
        client.poll(1).unwrap();
        assert_eq!(descriptor_count(&attachment_identity), 2);
        if invalid_bytes == 0 {
            drop(server);
        } else {
            let mut bytes = vec![0; invalid_bytes];
            if invalid_bytes == size_of::<ImageAdd>() {
                bytes.copy_from_slice(bytes_of(&image));
            }
            send(&server, &bytes, &vec![attachment.as_fd(); ancillary_count]);
        }

        assert_eq!(client.poll(2).unwrap_err().kind(), expected);
        assert!(!client.connected());
        assert!(client.hello.is_none() && client.frame.is_none() && client.images.is_empty());
        assert_eq!(client.frame_received_ns, 0);
        assert_eq!(client.take_removals(), [5]);
        assert_eq!(descriptor_count(&attachment_identity), 1);
    }
}
