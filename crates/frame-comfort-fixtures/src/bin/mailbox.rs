#[cfg(target_os = "linux")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use serde_json::{Value, json};
    use std::{
        env, fs, io,
        net::TcpListener,
        path::PathBuf,
        thread,
        time::{Duration, Instant},
    };
    use tungstenite::Message;

    let mut args = env::args_os().skip(1);
    let port_file = PathBuf::from(args.next().ok_or("expected a port output path")?);
    let status = PathBuf::from(args.next().ok_or("expected an app status path")?);
    let fallback = args
        .next()
        .map(|value| value.to_string_lossy().parse::<u64>())
        .transpose()?
        .map(Duration::from_secs);
    // A fixed push schedule, the same for every build under comparison:
    // "PUSHES,HOLD_MS,REST_MS,LEAD_MS". Without it, push once when the app
    // reports a standby snapshot.
    let schedule = match env::var("FRAME_COMFORT_MAILBOX_SCHEDULE") {
        Ok(text) => {
            let values = text
                .split(',')
                .map(|value| value.trim().parse::<u64>())
                .collect::<Result<Vec<_>, _>>()?;
            let [pushes, hold, rest, lead] = values[..] else {
                return Err("schedule is PUSHES,HOLD_MS,REST_MS,LEAD_MS".into());
            };
            Some((pushes, hold, rest, lead))
        }
        Err(_) => None,
    };
    let read_status = || {
        fs::read(&status)
            .ok()
            .and_then(|bytes| serde_json::from_slice::<Value>(&bytes).ok())
    };
    let listener = TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0))?;
    listener.set_nonblocking(true)?;
    fs::write(port_file, listener.local_addr()?.port().to_string())?;
    let start = Instant::now();
    let stream = loop {
        match listener.accept() {
            Ok((stream, _)) => break stream,
            Err(error)
                if error.kind() == io::ErrorKind::WouldBlock
                    && start.elapsed() < Duration::from_secs(8) =>
            {
                thread::sleep(Duration::from_millis(10));
            }
            Err(error) => return Err(error.into()),
        }
    };
    stream.set_read_timeout(Some(Duration::from_secs(2)))?;
    stream.set_write_timeout(Some(Duration::from_secs(2)))?;
    let mut mailbox = tungstenite::accept(stream)?;
    for _ in 0..3 {
        if !mailbox.read()?.is_text() {
            return Err(io::Error::other("expected a text mailbox subscription").into());
        }
    }
    let mut push = None;
    let mut next = Instant::now();
    let connected = next;
    let mut pushed = 0;
    let end = match schedule {
        Some((pushes, hold, rest, lead)) => {
            Duration::from_millis(lead + pushes * (hold + rest)) + connected.duration_since(start)
        }
        None => Duration::from_secs(25),
    };
    while next.duration_since(start) < end {
        let axis = if let Some((pushes, hold, rest, lead)) = schedule {
            let since = next.duration_since(connected).as_millis() as u64;
            let period = hold + rest;
            let index = since.saturating_sub(lead) / period;
            let holding = since >= lead && index < pushes && (since - lead) % period < hold;
            if holding && index == pushed {
                // Log the app's state at the moment each push begins.
                let state = read_status().unwrap_or_default();
                let rest_frame = &state["restFrame"];
                println!(
                    "{}",
                    json!({"push": index, "ms": since,
                        "standbyReady": rest_frame["standbyReady"],
                        "state": rest_frame["state"], "gameSource": state["gameSource"]})
                );
                pushed += 1;
                push = Some(next);
            }
            if holding { 0.8 } else { 0.0 }
        } else {
            if push.is_none()
                && (fallback.is_some_and(|wait| start.elapsed() >= wait)
                    || read_status()
                        .is_some_and(|state| state["restFrame"]["standbyReady"] == true))
            {
                push = Some(next);
            }
            if push.is_some_and(|at| next.duration_since(at) < Duration::from_secs(3)) {
                0.8
            } else {
                0.0
            }
        };
        if mailbox
            .send(Message::text(
                json!({
                    "type":"update_component_states",
                    "device":"/user/hand/left",
                    "components":{"/input/thumbstick/y":axis}
                })
                .to_string(),
            ))
            .is_err()
        {
            break;
        }
        next += Duration::from_millis(25);
        thread::sleep(next.saturating_duration_since(Instant::now()));
    }
    if push.is_none() {
        return Err("the overlay never reported a standby snapshot".into());
    }
    Ok(())
}

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("The SteamVR mailbox fixture requires Linux.");
    std::process::exit(1);
}
