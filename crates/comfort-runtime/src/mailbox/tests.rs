use super::*;
use std::{net::TcpListener, time::Instant};
use tungstenite::{
    WebSocket, accept,
    protocol::{
        frame::Frame,
        frame::coding::{Data, OpCode},
    },
};

fn input(device: &str, x: f32) -> Value {
    json!({"type":"update_component_states", "device":device,
        "components":{"/input/thumbstick/x":x,"/input/thumbstick/y":0.0}})
}

fn start(resubscribe: Duration) -> (Mailbox, TcpListener) {
    let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
    listener.set_nonblocking(true).unwrap();
    let mailbox = Mailbox::start(Options {
        port: listener.local_addr().unwrap().port(),
        reconnect: Duration::from_millis(150),
        resubscribe,
        ..Options::default()
    })
    .unwrap();
    (mailbox, listener)
}

fn receive_subscription(message: Message, kind: &str, device: &str, address: &str) {
    let message = message.into_text().unwrap();
    let payload = message.strip_prefix("mailbox_send input_server ").unwrap();
    let value: Value = serde_json::from_str(payload).unwrap();
    assert_eq!(value["type"], kind);
    assert_eq!(value["device_path"], device);
    assert_eq!(value["returnAddress"], address);
}

fn open(listener: &TcpListener) -> (WebSocket<TcpStream>, String) {
    let deadline = Instant::now() + Duration::from_secs(3);
    let stream = loop {
        match listener.accept() {
            Ok((stream, _)) => break stream,
            Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                assert!(Instant::now() < deadline, "mailbox did not connect");
                thread::sleep(Duration::from_millis(5));
            }
            Err(error) => panic!("mailbox accept failed: {error}"),
        }
    };
    stream.set_nonblocking(false).unwrap();
    stream
        .set_read_timeout(Some(Duration::from_secs(2)))
        .unwrap();
    stream
        .set_write_timeout(Some(Duration::from_secs(2)))
        .unwrap();
    let mut peer = accept(stream).unwrap();
    let address = format!("vrw_comfort/{}", std::process::id());
    assert_eq!(
        peer.read().unwrap().into_text().unwrap(),
        format!("mailbox_open {address}")
    );
    for device in DEVICES {
        receive_subscription(
            peer.read().unwrap(),
            "request_input_state_updates",
            device,
            &address,
        );
    }
    (peer, address)
}

fn wait_for(mailbox: &Mailbox, condition: impl Fn(&InputSnapshot) -> bool) -> InputSnapshot {
    let deadline = Instant::now() + Duration::from_secs(3);
    loop {
        let snapshot = mailbox.snapshot();
        if condition(&snapshot) {
            return snapshot;
        }
        assert!(
            Instant::now() < deadline,
            "unexpected mailbox state: {snapshot:?}"
        );
        thread::sleep(Duration::from_millis(5));
    }
}

#[test]
fn stick_reading_disarms_when_held_input_becomes_stale() {
    let mut state = InputSnapshot {
        connected: true,
        ..InputSnapshot::default()
    };
    assert!(state.apply(&input(DEVICES[0], 0.8), 100));
    let stale = Duration::from_millis(100);
    assert_eq!(state.read(Stick::Both, 0.15, stale, 99), Some([0.8, 0.0]));
    assert_eq!(state.read(Stick::Both, 0.15, stale, 100), Some([0.8, 0.0]));
    assert_eq!(state.read(Stick::Both, 0.15, stale, 200_000_000), None);
    assert!(state.read(Stick::Right, 0.15, stale, 200_000_000).is_some());
}

#[test]
fn mailbox_reconnects_without_stale_input_and_wakes_on_deadzone_crossing() {
    let (mailbox, listener) = start(Duration::from_secs(60));
    let (mut peer, _) = open(&listener);
    let text = input(DEVICES[0], 0.8).to_string();
    let (first, second) = text.as_bytes().split_at(text.len() / 2);
    peer.send(Message::Frame(Frame::message(
        first.to_vec(),
        OpCode::Data(Data::Text),
        false,
    )))
    .unwrap();
    peer.send(Message::Ping(b"ping".to_vec().into())).unwrap();
    assert_eq!(peer.read().unwrap(), Message::Pong(b"ping".to_vec().into()));
    peer.send(Message::Frame(Frame::message(
        second.to_vec(),
        OpCode::Data(Data::Continue),
        true,
    )))
    .unwrap();
    peer.send(Message::text(
        json!({"padding": "p".repeat(66_000)}).to_string(),
    ))
    .unwrap();
    let held = wait_for(&mailbox, |state| state.hands[0].seen && state.messages >= 2);
    assert!(held.connected);
    assert_eq!(held.hands[0].magnitude(), 0.8);

    let short = Duration::from_millis(50);
    let start = Instant::now();
    peer.send(Message::text(input(DEVICES[0], 0.7).to_string()))
        .unwrap();
    mailbox.wait_for_stick(short, 0.15);
    assert!(start.elapsed() >= short);

    let elapsed = thread::scope(|scope| {
        let wait = scope.spawn(|| {
            let start = Instant::now();
            mailbox.wait_for_stick(Duration::from_secs(3), 0.15);
            start.elapsed()
        });
        // The previous wait set the threshold, so the crossing must also be
        // delivered if it arrives before the waiter acquires the mutex.
        peer.send(Message::text(input(DEVICES[0], 0.0).to_string()))
            .unwrap();
        wait.join().unwrap()
    });
    assert!(
        elapsed < Duration::from_secs(3),
        "a crossing did not wake idle wait"
    );
    assert_eq!(mailbox.snapshot().hands[0].magnitude(), 0.0);

    peer.send(Message::text(input(DEVICES[0], 0.8).to_string()))
        .unwrap();
    wait_for(&mailbox, |state| state.hands[0].magnitude() == 0.8);
    peer.close(None).unwrap();
    wait_for(&mailbox, |state| !state.connected);
    let (_reconnected_peer, _) = open(&listener);
    let reconnected = wait_for(&mailbox, |state| state.connected);
    assert!(reconnected.hands.iter().all(|hand| !hand.seen));
}

#[test]
fn mailbox_renews_only_the_silent_controller_subscription() {
    let (mailbox, listener) = start(Duration::from_millis(750));
    let (mut peer, address) = open(&listener);
    peer.get_mut()
        .set_read_timeout(Some(Duration::from_millis(100)))
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(3);
    let cancel = loop {
        peer.send(Message::text(input(DEVICES[1], 0.8).to_string()))
            .unwrap();
        match peer.read() {
            Ok(message) => break message,
            Err(tungstenite::Error::Io(error))
                if matches!(
                    error.kind(),
                    io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut
                ) =>
            {
                assert!(
                    Instant::now() < deadline,
                    "silent controller subscription was not renewed"
                );
            }
            Err(error) => panic!("mailbox read failed: {error}"),
        }
    };
    receive_subscription(cancel, "cancel_input_state_updates", DEVICES[0], &address);
    receive_subscription(
        peer.read().unwrap(),
        "request_input_state_updates",
        DEVICES[0],
        &address,
    );
    let snapshot = wait_for(&mailbox, |state| state.resubscribes != 0);
    assert!(snapshot.hands[1].seen);
    assert!(!snapshot.hands[0].seen);
}
