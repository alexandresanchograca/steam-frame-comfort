//! Read-only SteamVR controller input. The worker only opens a mailbox and
//! requests or cancels component-state updates.
use comfort_core::settings::Stick;
use serde_json::{Value, json};
use std::{
    io,
    net::{Ipv4Addr, SocketAddr, SocketAddrV4, TcpStream},
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread::{self, JoinHandle},
    time::Duration,
};
use tungstenite::{Message, client::IntoClientRequest, protocol::WebSocketConfig};

const DEVICES: [&str; 2] = ["/user/hand/left", "/user/hand/right"];
const TIMEOUT: Duration = Duration::from_millis(200);

pub fn monotonic_ns() -> u64 {
    let now = rustix::time::clock_gettime(rustix::time::ClockId::Monotonic);
    (now.tv_sec as u64)
        .saturating_mul(1_000_000_000)
        .saturating_add(now.tv_nsec as u64)
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Hand {
    pub x: f32,
    pub y: f32,
    pub updated_ns: u64,
    pub seen: bool,
}

impl Hand {
    pub fn magnitude(self) -> f32 {
        if self.seen { self.x.hypot(self.y) } else { 0.0 }
    }
}

#[derive(Clone, Debug, Default)]
pub struct InputSnapshot {
    pub connected: bool,
    pub last_message_ns: u64,
    pub messages: u64,
    pub reconnects: u64,
    pub resubscribes: u64,
    pub hands: [Hand; 2],
    pub error: String,
}

impl InputSnapshot {
    fn apply(&mut self, message: &Value, now: u64) -> bool {
        self.last_message_ns = now;
        self.messages += 1;
        if message["type"] != "update_component_states" {
            return false;
        }
        let Some(hand) = message["device"].as_str().and_then(device_index) else {
            return false;
        };
        let stick = &mut self.hands[hand];
        let mut changed = false;
        for (x, y) in [
            ("/input/thumbstick/x", "/input/thumbstick/y"),
            ("/input/joystick/x", "/input/joystick/y"),
        ] {
            if let Some(value) = message["components"][x].as_f64() {
                stick.x = (value as f32).clamp(-1.0, 1.0);
                changed = true;
            }
            if let Some(value) = message["components"][y].as_f64() {
                stick.y = (value as f32).clamp(-1.0, 1.0);
                changed = true;
            }
        }
        if changed {
            stick.updated_ns = now;
            stick.seen = true;
        }
        changed
    }

    pub fn read(
        &self,
        stick: Stick,
        deadzone: f32,
        stale: Duration,
        now_ns: u64,
    ) -> Option<[f32; 2]> {
        let hands = self.hands.map(Hand::magnitude);
        let selected = match stick {
            Stick::Left => hands[0],
            Stick::Right => hands[1],
            Stick::Both => hands[0].max(hands[1]),
        };
        let silent = self.last_message_ns == 0
            || u128::from(now_ns.saturating_sub(self.last_message_ns)) > stale.as_nanos();
        (self.connected && (!silent || selected <= deadzone)).then_some(hands)
    }
}

fn device_index(device: &str) -> Option<usize> {
    DEVICES.iter().position(|&path| path == device)
}

struct State {
    snapshot: InputSnapshot,
    wake_threshold: f32,
    crossings: u64,
    seen_crossings: u64,
}

struct Shared {
    state: Mutex<State>,
    wake: Condvar,
    stop: AtomicBool,
}

pub struct Options {
    pub host: Ipv4Addr,
    pub port: u16,
    pub reconnect: Duration,
    pub resubscribe: Duration,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            host: Ipv4Addr::LOCALHOST,
            port: 27062,
            reconnect: Duration::from_secs(1),
            resubscribe: Duration::from_secs(3),
        }
    }
}

pub struct Mailbox {
    shared: Arc<Shared>,
    worker: Option<JoinHandle<()>>,
}

impl Mailbox {
    pub fn start(options: Options) -> io::Result<Self> {
        let shared = Arc::new(Shared {
            state: Mutex::new(State {
                snapshot: InputSnapshot::default(),
                wake_threshold: 2.0,
                crossings: 0,
                seen_crossings: 0,
            }),
            wake: Condvar::new(),
            stop: AtomicBool::new(false),
        });
        let worker_shared = Arc::clone(&shared);
        let worker = thread::Builder::new()
            .name("comfort-input".into())
            .spawn(move || worker(&worker_shared, options))?;
        Ok(Self {
            shared,
            worker: Some(worker),
        })
    }

    pub fn snapshot(&self) -> InputSnapshot {
        let mut state = self.shared.state.lock().unwrap();
        state.seen_crossings = state.crossings;
        state.snapshot.clone()
    }

    pub fn wait_for_stick(&self, timeout: Duration, threshold: f32) {
        let mut state = self.shared.state.lock().unwrap();
        state.wake_threshold = if threshold.is_finite() {
            threshold
        } else {
            2.0
        };
        drop(
            self.shared
                .wake
                .wait_timeout_while(state, timeout, |state| {
                    state.seen_crossings == state.crossings
                        && !self.shared.stop.load(Ordering::Relaxed)
                }),
        );
    }
}

impl Drop for Mailbox {
    fn drop(&mut self) {
        self.shared.stop.store(true, Ordering::Relaxed);
        if let Some(worker) = self.worker.take() {
            worker.thread().unpark();
            let _ = worker.join();
        }
    }
}

fn worker(shared: &Shared, options: Options) {
    while !shared.stop.load(Ordering::Relaxed) {
        let result = session(shared, &options);
        let mut state = shared.state.lock().unwrap();
        state.snapshot.connected = false;
        if let Err(error) = result {
            state.snapshot.reconnects += 1;
            state.snapshot.error = error.to_string();
        }
        state.crossings += 1;
        shared.wake.notify_all();
        drop(state);
        if !shared.stop.load(Ordering::Relaxed) {
            thread::park_timeout(options.reconnect);
        }
    }
}

fn session(shared: &Shared, options: &Options) -> Result<(), Box<dyn std::error::Error>> {
    let address = SocketAddrV4::new(options.host, options.port);
    let stream = TcpStream::connect_timeout(&SocketAddr::V4(address), TIMEOUT)?;
    stream.set_read_timeout(Some(TIMEOUT))?;
    stream.set_write_timeout(Some(TIMEOUT))?;
    stream.set_nodelay(true)?;
    let mut request = format!("ws://{address}/").into_client_request()?;
    request
        .headers_mut()
        .insert("Origin", format!("http://{address}").parse()?);
    let config = WebSocketConfig::default()
        .max_message_size(Some(1 << 20))
        .max_frame_size(Some(1 << 20));
    let (mut socket, _) = tungstenite::client::client_with_config(request, stream, Some(config))?;
    let name = format!("vrw_comfort/{}", std::process::id());
    socket.send(Message::text(format!("mailbox_open {name}")))?;
    let subscription = |device, kind| {
        Message::text(format!(
            "mailbox_send input_server {}",
            json!({"type": kind, "device_path": device, "returnAddress": name})
        ))
    };
    for device in DEVICES {
        socket.send(subscription(device, "request_input_state_updates"))?;
    }
    let mut last_by_device = [monotonic_ns(); 2];
    {
        let mut state = shared.state.lock().unwrap();
        state.snapshot.connected = true;
        state.snapshot.last_message_ns = monotonic_ns();
        state.snapshot.error.clear();
        state.snapshot.hands = [Hand::default(); 2];
    }
    let mut maintenance = monotonic_ns();
    while !shared.stop.load(Ordering::Relaxed) {
        let now = monotonic_ns();
        if now.saturating_sub(maintenance) >= 250_000_000 {
            maintenance = now;
            for (index, device) in DEVICES.into_iter().enumerate() {
                if u128::from(now.saturating_sub(last_by_device[index]))
                    >= options.resubscribe.as_nanos()
                {
                    socket.send(subscription(device, "cancel_input_state_updates"))?;
                    socket.send(subscription(device, "request_input_state_updates"))?;
                    last_by_device[index] = now;
                    shared.state.lock().unwrap().snapshot.resubscribes += 1;
                }
            }
        }
        let message = match socket.read() {
            Ok(Message::Text(text)) => serde_json::from_slice::<Value>(text.as_bytes()).ok(),
            Ok(Message::Binary(bytes)) => serde_json::from_slice::<Value>(&bytes).ok(),
            Ok(Message::Close(_)) | Err(tungstenite::Error::ConnectionClosed) => return Ok(()),
            Ok(Message::Ping(_)) => {
                socket.flush()?;
                continue;
            }
            Ok(_) => continue,
            Err(tungstenite::Error::Io(error))
                if matches!(
                    error.kind(),
                    io::ErrorKind::WouldBlock
                        | io::ErrorKind::TimedOut
                        | io::ErrorKind::Interrupted
                ) =>
            {
                continue;
            }
            Err(error) => return Err(error.into()),
        };
        let now = monotonic_ns();
        let mut state = shared.state.lock().unwrap();
        if let Some(message) = message {
            if let Some(hand) = message["device"].as_str().and_then(device_index) {
                last_by_device[hand] = now;
            }
            let before = state.snapshot.hands;
            let changed = state.snapshot.apply(&message, now);
            let threshold = state.wake_threshold;
            if changed
                && before.iter().zip(state.snapshot.hands).any(|(old, new)| {
                    (old.magnitude() > threshold) != (new.magnitude() > threshold)
                })
            {
                state.crossings += 1;
                shared.wake.notify_all();
            }
        } else {
            state.snapshot.last_message_ns = now;
        }
    }
    for device in DEVICES {
        let _ = socket.send(subscription(device, "cancel_input_state_updates"));
    }
    let _ = socket.send(Message::text("websocket_close"));
    Ok(())
}

#[cfg(test)]
mod tests;
