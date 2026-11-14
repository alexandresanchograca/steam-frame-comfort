use comfort_protocol::{DecodeError, Frame, Hello, ImageAdd, MAX_PACKET_SIZE, Packet, decode};
use rustix::{
    fd::OwnedFd,
    io::Errno,
    net::{
        AddressFamily, RecvAncillaryBuffer, RecvAncillaryMessage, RecvFlags, ReturnFlags,
        SocketAddrUnix, SocketFlags, SocketType, connect, recvmsg, socket_with, sockopt,
    },
};
use std::{
    collections::BTreeMap,
    io::{self, IoSliceMut},
    mem::MaybeUninit,
    path::Path,
};

#[cfg(test)]
mod tests;

pub struct SourceImage {
    pub description: ImageAdd,
    pub fd: OwnedFd,
    pub revision: u64,
}

/// The connection owns all received descriptors until replacement, removal or disconnect.
#[derive(Default)]
pub struct SourceClient {
    socket: Option<OwnedFd>,
    pub hello: Option<Hello>,
    pub images: BTreeMap<u64, SourceImage>,
    pub frame: Option<Frame>,
    pub frame_received_ns: u64,
    pub protocol_errors: u64,
    removals: Vec<u64>,
    revision: u64,
}

impl SourceClient {
    pub fn connected(&self) -> bool {
        self.socket.is_some()
    }
    pub fn connect(&mut self, path: &Path) -> io::Result<()> {
        self.disconnect();
        let addr = SocketAddrUnix::new(path)?;
        let socket = socket_with(
            AddressFamily::UNIX,
            SocketType::SEQPACKET,
            SocketFlags::NONBLOCK | SocketFlags::CLOEXEC,
            None,
        )?;
        connect(&socket, &addr)?;
        if sockopt::socket_peercred(&socket)?.uid != rustix::process::getuid() {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "source socket owned by another user",
            ));
        }
        self.socket = Some(socket);
        Ok(())
    }

    pub fn disconnect(&mut self) {
        self.socket = None;
        self.removals.extend(self.images.keys().copied());
        self.images.clear();
        self.hello = None;
        self.frame = None;
        self.frame_received_ns = 0;
    }

    pub fn take_removals(&mut self) -> Vec<u64> {
        std::mem::take(&mut self.removals)
    }

    pub fn poll(&mut self, now_ns: u64) -> io::Result<()> {
        let result = self.drain(now_ns);
        if result.is_err() {
            self.disconnect();
        }
        result
    }

    fn drain(&mut self, now_ns: u64) -> io::Result<()> {
        let socket = self.socket.as_ref().ok_or(io::ErrorKind::NotConnected)?;
        loop {
            let mut bytes = [0; MAX_PACKET_SIZE];
            let mut ancillary = [MaybeUninit::uninit(); rustix::cmsg_space!(ScmRights(4))];
            let mut control = RecvAncillaryBuffer::new(&mut ancillary);
            let message = match recvmsg(
                socket,
                &mut [IoSliceMut::new(&mut bytes)],
                &mut control,
                RecvFlags::CMSG_CLOEXEC,
            ) {
                Err(Errno::AGAIN) => return Ok(()),
                Err(Errno::INTR) => continue,
                other => other?,
            };
            if message.bytes == 0 {
                return Err(io::ErrorKind::UnexpectedEof.into());
            }

            // Dropping the ancillary buffer closes descriptors we do not consume.
            let packet = if message
                .flags
                .intersects(ReturnFlags::TRUNC | ReturnFlags::CTRUNC)
            {
                Err(DecodeError::Header)
            } else {
                decode(&bytes[..message.bytes])
            };
            match packet {
                Ok(Packet::Hello(hello)) => self.hello = Some(hello),
                Ok(Packet::Frame(frame)) => {
                    self.frame = Some(frame);
                    self.frame_received_ns = now_ns;
                }
                Ok(Packet::Remove(image)) => {
                    self.images.remove(&image.id);
                    self.removals.push(image.id);
                }
                Ok(Packet::Add(description)) => {
                    let mut descriptors = control
                        .drain()
                        .filter_map(|message| match message {
                            RecvAncillaryMessage::ScmRights(fds) => Some(fds),
                            _ => None,
                        })
                        .flatten();
                    match (descriptors.next(), descriptors.next()) {
                        (Some(fd), None) => {
                            self.revision += 1;
                            self.images.insert(
                                description.id,
                                SourceImage {
                                    description,
                                    fd,
                                    revision: self.revision,
                                },
                            );
                        }
                        _ => self.protocol_errors += 1,
                    }
                }
                Err(DecodeError::FrameCount) => self.protocol_errors += 1,
                Err(DecodeError::Header) => {
                    self.protocol_errors += 1;
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "invalid source message",
                    ));
                }
            }
        }
    }
}
