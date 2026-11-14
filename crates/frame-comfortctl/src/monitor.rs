use super::{Paths, app_status, pids, proc_ids, proc_text};
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs, io,
    io::{BufWriter, Write},
    path::PathBuf,
    thread,
    time::{Duration, Instant},
};

const WATCHED: [&str; 3] = ["vrcompositor", "vrw-comfort", "vrlink"];
const GPU_KEYS: [&str; 4] = ["total", "vrcompositor", "vrw-comfort", "vrlink"];
const APP_KEYS: [&str; 9] = [
    "rule",
    "blend",
    "rendered",
    "renderFailures",
    "activeFrames",
    "gpuMsP95",
    "error",
    "gameSource",
    "restFrame",
];

#[derive(Clone, Copy)]
struct ProcessSample {
    pid: u32,
    cpu_seconds: f64,
    rss_mib: f64,
    fds: Option<usize>,
}

fn process_sample(pid: u32) -> Option<ProcessSample> {
    let stat = proc_text(pid, "stat")?;
    // The comm field can contain spaces or parentheses. Count fields after it.
    let fields: Vec<_> = stat.rsplit_once(')')?.1.split_whitespace().collect();
    let user = fields.get(11)?.parse::<f64>().ok()?;
    let system = fields.get(12)?.parse::<f64>().ok()?;
    let pages = fields.get(21)?.parse::<f64>().ok()?;
    Some(ProcessSample {
        pid,
        cpu_seconds: (user + system) / rustix::param::clock_ticks_per_second() as f64,
        rss_mib: pages * rustix::param::page_size() as f64 / 2_f64.powi(20),
        fds: fs::read_dir(format!("/proc/{pid}/fd"))
            .ok()
            .map(Iterator::count),
    })
}

fn drm_fds() -> Vec<(u32, PathBuf)> {
    let mut found = Vec::new();
    for pid in proc_ids() {
        for entry in fs::read_dir(format!("/proc/{pid}/fd"))
            .into_iter()
            .flatten()
            .flatten()
        {
            if fs::read_link(entry.path()).is_ok_and(|link| link.starts_with("/dev/dri")) {
                found.push((
                    pid,
                    PathBuf::from(format!("/proc/{pid}/fdinfo")).join(entry.file_name()),
                ));
            }
        }
    }
    found
}

fn gpu_clients(fds: &[(u32, PathBuf)]) -> BTreeMap<u64, (u32, u64)> {
    let mut clients = BTreeMap::new();
    for (pid, path) in fds {
        let Ok(text) = fs::read_to_string(path) else {
            continue;
        };
        let mut id = None;
        let mut engine = None;
        for line in text.lines() {
            if let Some((key, value)) = line.split_once(':') {
                match key {
                    "drm-client-id" => id = value.trim().parse::<u64>().ok(),
                    "drm-engine-gpu" => {
                        engine = value
                            .split_whitespace()
                            .next()
                            .and_then(|v| v.parse::<u64>().ok())
                    }
                    _ => {}
                }
            }
        }
        if let (Some(id), Some(engine)) = (id, engine) {
            let entry = clients.entry(id).or_insert((*pid, engine));
            if engine > entry.1 {
                *entry = (*pid, engine);
            }
        }
    }
    clients
}

fn gpu_frequency() -> Option<f64> {
    fs::read_dir("/sys/class/devfreq")
        .ok()?
        .flatten()
        .filter(|entry| entry.file_name().to_string_lossy().contains("gpu"))
        .find_map(|entry| {
            fs::read_to_string(entry.path().join("cur_freq"))
                .ok()?
                .trim()
                .parse::<f64>()
                .ok()
        })
        .map(|hz| hz / 1e6)
}

fn counter(app: Option<&Value>, key: &str) -> u64 {
    app.and_then(|value| value[key].as_u64()).unwrap_or(0)
}

fn round(value: f64, digits: i32) -> f64 {
    let factor = 10_f64.powi(digits);
    (value * factor).round_ties_even() / factor
}

struct GpuSample {
    active: bool,
    pct: BTreeMap<String, f64>,
}

fn stats(samples: &[GpuSample], name: &str, active: Option<bool>) -> Value {
    let mut values: Vec<_> = samples
        .iter()
        .filter(|sample| active.is_none_or(|active| sample.active == active))
        .map(|sample| sample.pct.get(name).copied().unwrap_or(0.0))
        .collect();
    if values.is_empty() {
        return Value::Null;
    }
    values.sort_by(f64::total_cmp);
    let index = ((0.95 * values.len() as f64 + 0.5).round_ties_even() as usize)
        .saturating_sub(1)
        .min(values.len() - 1);
    json!({"mean": round(values.iter().sum::<f64>() / values.len() as f64, 2),
        "p95": round(values[index], 2), "samples": values.len()})
}

pub(super) fn monitor(
    paths: &Paths,
    seconds: Duration,
    interval: Duration,
    out: Option<PathBuf>,
    verbose: bool,
) -> io::Result<Value> {
    let mut out = out.map(fs::File::create).transpose()?.map(BufWriter::new);
    let mut fds = drm_fds();
    let mut fds_at = Instant::now();
    let mut previous_clients = gpu_clients(&fds);
    let mut names = BTreeMap::new();
    let mut previous_processes: BTreeMap<&str, ProcessSample> = BTreeMap::new();
    let mut first_processes = BTreeMap::new();
    let mut last_processes = BTreeMap::new();
    let (mut previous_app, mut first_app, mut last_app) = (None, None, None);
    let mut rules = BTreeSet::new();
    let mut samples = Vec::new();
    let start = Instant::now();
    let mut previous_wall = start;
    let mut last_time = 0.0;

    while start.elapsed() < seconds {
        thread::sleep(interval);
        if fds_at.elapsed() > Duration::from_secs(10) {
            fds = drm_fds();
            fds_at = Instant::now();
        }
        let clients = gpu_clients(&fds);
        let wall = Instant::now();
        let elapsed = wall.duration_since(previous_wall).as_secs_f64();
        let mut gpu = BTreeMap::new();
        for (&id, &(pid, ns)) in &clients {
            if let Some(&(_, previous_ns)) = previous_clients.get(&id) {
                let name = names.entry(pid).or_insert_with(|| {
                    proc_text(pid, "comm")
                        .unwrap_or_else(|| "?".into())
                        .trim()
                        .to_owned()
                });
                let name = if WATCHED.contains(&name.as_str()) {
                    name.as_str()
                } else {
                    "other"
                };
                *gpu.entry(name.to_owned()).or_insert(0.0) +=
                    ns.saturating_sub(previous_ns) as f64 / 1e9 / elapsed * 100.0;
            }
        }
        gpu.insert("total".into(), gpu.values().sum());
        for value in gpu.values_mut() {
            *value = round(*value, 2);
        }
        previous_clients = clients;
        previous_wall = wall;

        let mut processes = BTreeMap::new();
        let mut process_json = BTreeMap::new();
        for name in WATCHED {
            let Some(sample) = pids(name).first().and_then(|&pid| process_sample(pid)) else {
                continue;
            };
            let cpu = previous_processes
                .get(name)
                .filter(|previous| previous.pid == sample.pid)
                .map(|previous| {
                    round(
                        (sample.cpu_seconds - previous.cpu_seconds) / elapsed * 100.0,
                        2,
                    )
                });
            process_json.insert(
                name,
                json!({"cpuPct": cpu, "rssMiB": round(sample.rss_mib, 1), "fds": sample.fds}),
            );
            first_processes.entry(name).or_insert(sample);
            last_processes.insert(name, sample);
            processes.insert(name, sample);
        }
        previous_processes = processes;

        let app = app_status(&paths.status).filter(Value::is_object);
        let active = app.is_some()
            && previous_app.is_some()
            && counter(app.as_ref(), "activeFrames")
                > counter(previous_app.as_ref(), "activeFrames");
        let app_fields = app.as_ref().map(|app| {
            APP_KEYS
                .into_iter()
                .map(|key| (key, app[key].clone()))
                .collect::<BTreeMap<_, _>>()
        });
        if let Some(app) = app {
            if let Some(rule) = app["rule"]
                .as_str()
                .or_else(|| app["restFrame"]["rule"].as_str())
            {
                rules.insert(rule.to_owned());
            }
            first_app.get_or_insert_with(|| app.clone());
            last_app = Some(app.clone());
            previous_app = Some(app);
        }
        last_time = round(wall.duration_since(start).as_secs_f64(), 1);
        let record = json!({"t": last_time, "freqMHz": gpu_frequency(), "comfortActive": active,
            "gpuPct": gpu, "processes": process_json, "app": app_fields});
        samples.push(GpuSample { active, pct: gpu });
        if let Some(out) = &mut out {
            writeln!(out, "{record}")?;
            out.flush()?;
        }
        if verbose {
            println!("{record}");
            io::stdout().flush()?;
        }
    }

    let gpu: BTreeMap<_, _> = GPU_KEYS
        .into_iter()
        .map(|name| {
            (
                name,
                json!({
                    "all": stats(&samples, name, None),
                    "comfortActive": stats(&samples, name, Some(true)),
                    "comfortIdle": stats(&samples, name, Some(false)),
                }),
            )
        })
        .collect();
    let processes: BTreeMap<_, _> = first_processes
        .iter()
        .map(|(&name, first)| {
            let last = last_processes[name];
            (
                name,
                json!({"pidStable": first.pid == last.pid, "fds": [first.fds, last.fds],
            "rssMiB": [round(first.rss_mib, 1), round(last.rss_mib, 1)]}),
            )
        })
        .collect();
    let app = first_app.zip(last_app).map(|(first, last)| {
        let delta =
            |key| i128::from(counter(Some(&last), key)) - i128::from(counter(Some(&first), key));
        json!({"rendered": delta("rendered"), "renderFailures": delta("renderFailures"),
            "activeFrames": delta("activeFrames"), "rules": rules})
    });
    let summary = json!({"summary": true, "seconds": last_time,
        "activeSamples": samples.iter().filter(|sample| sample.active).count(),
        "gpuPct": gpu, "processes": processes, "app": app});
    if let Some(out) = &mut out {
        writeln!(out, "{summary}")?;
        out.flush()?;
    }
    Ok(summary)
}
