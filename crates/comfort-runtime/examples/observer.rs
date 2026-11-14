//! Check the Rust client against a running observer on a private fixture socket.
#[cfg(target_os = "linux")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use comfort_runtime::source_client::SourceClient;
    use std::{
        path::PathBuf,
        time::{Duration, Instant},
    };

    let mut args = std::env::args_os().skip(1);
    let socket = args
        .next()
        .map(PathBuf::from)
        .ok_or("usage: observer SOCKET [TRACE_SECONDS]")?;
    let trace = args
        .next()
        .map(|duration| duration.to_string_lossy().parse::<u64>())
        .transpose()?;
    let mut previous = None;
    let mut received = false;
    let mut client = SourceClient::default();
    client.connect(&socket)?;
    let start = Instant::now();
    while start.elapsed() < Duration::from_secs(trace.unwrap_or(5)) {
        client.poll(start.elapsed().as_nanos() as u64)?;
        if let (Some(hello), Some(frame)) = (client.hello, client.frame)
            && !client.images.is_empty()
        {
            if trace.is_some() {
                let shape = |ids: &[u64], eyes: &[comfort_protocol::EyeUse]| {
                    ids.iter()
                        .zip(eyes)
                        .filter_map(|(&id, eye)| {
                            let image = client.images.get(&id)?.description;
                            Some((
                                id,
                                image.format,
                                image.width,
                                image.height,
                                image.array_layers,
                                image.exported,
                                eye.eyes,
                                eye.slot,
                            ))
                        })
                        .collect::<Vec<_>>()
                };
                let acquired = shape(&frame.ids[..frame.count as usize], &frame.eyes);
                let touched = shape(
                    &frame.touched_ids[..frame.touched_count as usize],
                    &frame.touched_eyes,
                );
                let state = (frame.truncated, frame.touched_truncated, acquired, touched);
                if previous.as_ref() != Some(&state) {
                    println!(
                        "{:.3} seq={} {state:?}",
                        start.elapsed().as_secs_f64(),
                        frame.sequence
                    );
                    previous = Some(state);
                }
                received = true;
                std::thread::sleep(Duration::from_millis(10));
                continue;
            }
            println!(
                "Observer v{}, PID {}, frame {}, {} imported descriptors",
                hello.header.version,
                hello.pid,
                frame.sequence,
                client.images.len()
            );
            return Ok(());
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    if received {
        Ok(())
    } else {
        Err("no image and frame received from the observer".into())
    }
}

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("The observer connection requires Linux.");
    std::process::exit(1);
}
