use super::observe;
use comfort_protocol::{Header, Hello, ImageRemove};
use rustix::{
    event::{EventfdFlags, PollFd, PollFlags, Timespec, eventfd, poll},
    net::{
        AddressFamily, SendAncillaryBuffer, SendAncillaryMessage, SendFlags, SocketAddrUnix,
        SocketFlags, SocketType, accept_with, bind, listen, sendmsg, socket_with, sockopt,
    },
};
use std::{
    collections::BTreeMap,
    fs,
    io::{self, IoSlice},
    mem::MaybeUninit,
    os::{
        fd::{AsFd, BorrowedFd, OwnedFd},
        unix::fs::PermissionsExt,
    },
    path::PathBuf,
    sync::{
        Arc, Mutex, OnceLock,
        atomic::{AtomicBool, Ordering},
    },
    thread,
};

struct Worker {
    stop: Arc<AtomicBool>,
    thread: thread::JoinHandle<()>,
}

static WORKER: Mutex<Option<Worker>> = Mutex::new(None);
static PATH: OnceLock<Option<PathBuf>> = OnceLock::new();
// Wakes the socket thread. Kept open for the life of the process, so the
// compositor's hooks never write to a descriptor that may have been reused.
static WAKE: OnceLock<Option<OwnedFd>> = OnceLock::new();
static CONNECTED: AtomicBool = AtomicBool::new(false);

fn wake_fd() -> Option<BorrowedFd<'static>> {
    WAKE.get_or_init(|| eventfd(0, EventfdFlags::CLOEXEC | EventfdFlags::NONBLOCK).ok())
        .as_ref()
        .map(AsFd::as_fd)
}

fn signal(wake: BorrowedFd<'_>) {
    let _ = rustix::io::write(wake, &1_u64.to_ne_bytes());
}

/// Wake the socket thread because a frame or the image set changed. While no
/// app is connected this is one atomic load and no system call.
pub fn notify() {
    if CONNECTED.load(Ordering::Acquire)
        && let Some(Some(wake)) = WAKE.get()
    {
        signal(wake.as_fd());
    }
}

/// Block until `fd` is readable or hung up, the thread is woken, or a second
/// passes. Without an eventfd, fall back to polling every 2 ms.
fn wait(fd: BorrowedFd<'_>, wake: Option<BorrowedFd<'_>>) -> io::Result<()> {
    let timeout = if wake.is_some() {
        Timespec {
            tv_sec: 1,
            tv_nsec: 0,
        }
    } else {
        Timespec {
            tv_sec: 0,
            tv_nsec: 2_000_000,
        }
    };
    let mut fds = [
        PollFd::from_borrowed_fd(fd, PollFlags::IN),
        PollFd::from_borrowed_fd(wake.unwrap_or(fd), PollFlags::IN),
    ];
    match poll(&mut fds, Some(&timeout)) {
        Ok(_) | Err(rustix::io::Errno::INTR) => Ok(()),
        Err(error) => Err(error.into()),
    }
}

fn drain(wake: Option<BorrowedFd<'_>>) {
    if let Some(wake) = wake {
        let _ = rustix::io::read(wake, &mut [0_u8; 8]);
    }
}

pub fn socket_path() -> Option<PathBuf> {
    PATH.get_or_init(|| {
        let expected =
            std::env::var("VRW_COMFORT_SOURCE_COMM").unwrap_or_else(|_| "vrcompositor".into());
        if !expected.is_empty()
            && !fs::read_to_string("/proc/self/comm").is_ok_and(|name| name.trim_end() == expected)
        {
            return None;
        }
        std::env::var_os("VRW_COMFORT_SOURCE_SOCKET")
            .map(PathBuf::from)
            .filter(|path| path.is_absolute())
            .or_else(|| {
                std::env::var_os("XDG_RUNTIME_DIR")
                    .map(PathBuf::from)
                    .filter(|path| path.is_absolute())
                    .map(|path| path.join("vrw-comfort-source.sock"))
            })
    })
    .clone()
}

pub fn start(path: PathBuf, device_uuid: [u8; 16], driver_uuid: [u8; 16]) {
    let mut worker = WORKER.lock().unwrap_or_else(|error| error.into_inner());
    if worker.is_some() {
        return;
    }
    let stop = Arc::new(AtomicBool::new(false));
    let flag = Arc::clone(&stop);
    let hello = Hello {
        header: Header::observer::<Hello>(1),
        pid: std::process::id(),
        reserved: 0,
        device_uuid,
        driver_uuid,
    };
    if let Ok(thread) = thread::Builder::new()
        .name("comfort-source".into())
        .spawn(move || {
            if let Err(error) = serve(path, hello, &flag) {
                eprintln!("Frame Comfort source socket: {error}");
            }
        })
    {
        *worker = Some(Worker { stop, thread });
    }
}

pub fn stop() {
    if let Some(worker) = WORKER
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .take()
    {
        worker.stop.store(true, Ordering::Release);
        if let Some(wake) = wake_fd() {
            signal(wake);
        }
        let _ = worker.thread.join();
    }
}

fn send<T: bytemuck::Pod>(
    client: BorrowedFd<'_>,
    packet: &T,
    fd: Option<BorrowedFd<'_>>,
) -> io::Result<()> {
    let bytes = bytemuck::bytes_of(packet);
    let mut storage = [MaybeUninit::uninit(); rustix::cmsg_space!(ScmRights(1))];
    let mut ancillary = SendAncillaryBuffer::new(&mut storage);
    let fds = fd.map(|fd| [fd]);
    if let Some(fds) = &fds {
        ancillary.push(SendAncillaryMessage::ScmRights(fds));
    }
    let size = sendmsg(
        client,
        &[IoSlice::new(bytes)],
        &mut ancillary,
        SendFlags::NOSIGNAL,
    )?;
    if size != bytes.len() {
        return Err(io::ErrorKind::WriteZero.into());
    }
    Ok(())
}

fn serve(path: PathBuf, hello: Hello, stop: &AtomicBool) -> io::Result<()> {
    use std::os::unix::fs::{FileTypeExt, MetadataExt};
    let address = SocketAddrUnix::new(&path)?;
    let listener = socket_with(
        AddressFamily::UNIX,
        SocketType::SEQPACKET,
        SocketFlags::NONBLOCK | SocketFlags::CLOEXEC,
        None,
    )?;
    if let Ok(metadata) = path.symlink_metadata() {
        if !metadata.file_type().is_socket() || metadata.uid() != rustix::process::getuid().as_raw()
        {
            return Err(io::Error::other(
                "source path is not a socket owned by this user",
            ));
        }
        let probe = socket_with(
            AddressFamily::UNIX,
            SocketType::SEQPACKET,
            SocketFlags::NONBLOCK | SocketFlags::CLOEXEC,
            None,
        )?;
        match rustix::net::connect(probe, &address) {
            Err(rustix::io::Errno::CONNREFUSED) => fs::remove_file(&path)?,
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::AddrInUse,
                    "source socket is active",
                ));
            }
        }
    }
    bind(&listener, &address)?;
    struct SocketPath(PathBuf);
    impl Drop for SocketPath {
        fn drop(&mut self) {
            let _ = fs::remove_file(&self.0);
        }
    }
    let path = SocketPath(path);
    fs::set_permissions(&path.0, fs::Permissions::from_mode(0o600))?;
    listen(&listener, 1)?;
    let wake = wake_fd();
    while !stop.load(Ordering::Acquire) {
        wait(listener.as_fd(), wake)?;
        drain(wake);
        let client = match accept_with(&listener, SocketFlags::NONBLOCK | SocketFlags::CLOEXEC) {
            Ok(client) => client,
            Err(rustix::io::Errno::AGAIN | rustix::io::Errno::INTR) => continue,
            Err(error) => return Err(error.into()),
        };
        if sockopt::socket_peercred(&client).is_ok_and(|cred| cred.uid == rustix::process::getuid())
            && send(client.as_fd(), &hello, None).is_ok()
        {
            struct Connected;
            impl Drop for Connected {
                fn drop(&mut self) {
                    CONNECTED.store(false, Ordering::Release);
                }
            }
            // Changes made from here on wake the thread. Earlier ones are in
            // the first snapshot.
            CONNECTED.store(true, Ordering::Release);
            let _connected = Connected;
            let _ = publish(client.as_fd(), stop, wake);
        }
    }
    Ok(())
}

fn publish(
    client: BorrowedFd<'_>,
    stop: &AtomicBool,
    wake: Option<BorrowedFd<'_>>,
) -> io::Result<()> {
    let mut sent = BTreeMap::new();
    let mut sequence = 0;
    let mut previous_epoch = None;
    while !stop.load(Ordering::Acquire) {
        // Clear the wake-up before reading the state it reports.
        drain(wake);
        let mut unexpected = [0_u8; 1];
        match rustix::net::recv(client, &mut unexpected, rustix::net::RecvFlags::DONTWAIT) {
            Err(rustix::io::Errno::AGAIN | rustix::io::Errno::INTR) => {}
            _ => return Ok(()),
        }
        if let Some(update) = observe::snapshot(previous_epoch, sequence) {
            if let Some(images) = update.images {
                for &id in sent.keys() {
                    if !images.contains_key(&id) {
                        let remove = ImageRemove {
                            header: Header::observer::<ImageRemove>(3),
                            id,
                        };
                        send(client, &remove, None)?;
                    }
                }
                sent.retain(|id, _| images.contains_key(id));
                for (id, image) in images {
                    if sent.get(&id) != Some(&image.description) {
                        send(client, &image.description, Some(image.fd.as_fd()))?;
                        sent.insert(id, image.description);
                    }
                }
            }
            if let Some(frame) = update.frame {
                send(client, &frame, None)?;
                sequence = frame.sequence;
            }
            previous_epoch = Some(update.epoch);
        }
        wait(client, wake)?;
    }
    Ok(())
}
