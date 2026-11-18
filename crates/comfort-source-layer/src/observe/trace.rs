//! Opt-in compositor trace for diagnosing how a source image is sampled.
//!
//! With `VRW_COMFORT_SOURCE_TRACE` set to a file path, each draw that the
//! observer labels with an eye also records the bound pipeline's shader
//! hashes and the bytes of the uniform and storage buffers bound to the
//! graphics descriptor sets. A rate-limited sample is appended to the file as
//! JSON lines. Without the variable nothing here runs on the render path.
use super::*;
use std::{
    fs::{File, OpenOptions},
    io::Write,
    sync::atomic::{AtomicU64, Ordering},
};

/// Bytes copied per buffer binding. The compositor's largest distortion block
/// is 2800 bytes.
const MAX_BYTES: u64 = 4096;
const MAX_RECORDS: u64 = 400;
const INTERVAL_NS: u64 = 1_000_000_000;

static FILE: OnceLock<Option<Mutex<File>>> = OnceLock::new();
static RECORDS: AtomicU64 = AtomicU64::new(0);
static LAST_NS: AtomicU64 = AtomicU64::new(0);

fn file() -> Option<&'static Mutex<File>> {
    FILE.get_or_init(|| {
        let path = std::env::var_os("VRW_COMFORT_SOURCE_TRACE")?;
        OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .ok()
            .map(Mutex::new)
    })
    .as_ref()
}

pub(super) fn enabled() -> bool {
    file().is_some()
}

pub(super) fn write_line(line: &str) {
    if let Some(file) = file() {
        let mut file = file.lock().unwrap_or_else(|error| error.into_inner());
        let _ = writeln!(file, "{line}");
    }
}

/// Claim the next sample slot, at most once per interval.
pub(super) fn due(now: u64) -> bool {
    if RECORDS.load(Ordering::Relaxed) >= MAX_RECORDS {
        return false;
    }
    let last = LAST_NS.load(Ordering::Relaxed);
    if now.saturating_sub(last) < INTERVAL_NS
        || LAST_NS
            .compare_exchange(last, now, Ordering::Relaxed, Ordering::Relaxed)
            .is_err()
    {
        return false;
    }
    RECORDS.fetch_add(1, Ordering::Relaxed);
    true
}

pub(super) fn hash(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, &byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

#[derive(Clone)]
pub(super) struct TracedDraw {
    pub eye: usize,
    pub labels: Vec<(u64, u32, u32)>,
    pub stages: [u64; 2],
    pub viewport: [f32; 4],
    pub area: [i32; 4],
    pub buffers: Vec<super::BoundBuffer>,
}

impl State {
    pub(super) fn trace_submit(&self, device: vk::Device, draws: &[TracedDraw], now: u64) {
        use std::fmt::Write as _;
        let mut line = format!("{{\"t\":{now},\"draws\":[");
        for (index, draw) in draws.iter().enumerate() {
            if index != 0 {
                line.push(',');
            }
            let _ = write!(
                line,
                "{{\"eye\":{},\"labels\":{:?},\"vs\":\"{:016x}\",\"fs\":\"{:016x}\",\"viewport\":{:?},\"area\":{:?},\"buffers\":[",
                draw.eye,
                draw.labels
                    .iter()
                    .map(|&(id, binding, layer)| [id, u64::from(binding), u64::from(layer)])
                    .collect::<Vec<_>>(),
                draw.stages[0],
                draw.stages[1],
                draw.viewport,
                draw.area,
            );
            for (index, bound) in draw.buffers.iter().enumerate() {
                if index != 0 {
                    line.push(',');
                }
                let words = self.read_words(device, bound, 0, bound.range.min(MAX_BYTES));
                let _ = write!(
                    line,
                    "{{\"set\":{},\"binding\":{},\"element\":{},\"type\":{},\"offset\":{},\"dynamic\":{},\"range\":{},\"mapped\":{},\"words\":[",
                    bound.set,
                    bound.binding,
                    bound.element,
                    bound.kind,
                    bound.offset,
                    bound.dynamic.map_or(-1, i64::from),
                    bound.range,
                    words.is_some(),
                );
                for (index, word) in words.iter().flatten().enumerate() {
                    if index != 0 {
                        line.push(',');
                    }
                    let _ = write!(line, "{word}");
                }
                line.push_str("]}");
            }
            line.push_str("]}");
        }
        line.push_str("]}");
        write_line(&line);
    }
}
