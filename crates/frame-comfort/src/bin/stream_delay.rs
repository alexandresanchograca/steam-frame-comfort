//! Measure a stream's render-to-report delay from `--probe-stream-motion`.
//!
//! For each recorded frame, the strip's sideways shift since a frame about
//! `--gap-ms` earlier is compared with the head's yaw change over the same
//! interval, moved back by a trial delay. The delay with the strongest
//! correlation is the value for `restFrameStreamPoseOffsetMs`. Turn the head
//! slowly (about 20 to 60°/s) while recording: fast turns leave too little
//! overlap between strips, and the surviving matches bias the result.
use clap::Parser;
use serde_json::{Value, json};
use std::{
    error::Error,
    fs,
    io::{self, BufRead, BufReader},
    path::PathBuf,
};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

/// The Frame's raw eye tangents, used when the recording has none.
const FRAME_TANGENTS: [[f64; 4]; 2] = [
    [-1.66486, 1.21300, -1.72748, 1.17198],
    [-1.21868, 1.65399, -1.72807, 1.17120],
];
const SHIFT_LIMIT: i32 = 40;
const MIN_PEAK: f64 = 0.5;
const MAX_EYE_DISAGREEMENT_DEG: f64 = 0.6;

#[derive(Parser)]
#[command(about = "Stream render-to-report delay from --probe-stream-motion output")]
struct Options {
    /// Strip file written by --probe-dump.
    strips: PathBuf,
    /// The probe's stdout: poses and eye tangents as JSON lines.
    poses: PathBuf,
    /// Time between the two frames of each compared pair.
    #[arg(long, default_value_t = 50.0)]
    gap_ms: f64,
}

struct Strips {
    width: usize,
    rows: usize,
    frames: Vec<(u64, [Vec<f64>; 2])>,
}

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

fn read_strips(bytes: &[u8]) -> io::Result<Strips> {
    if bytes.get(..8) != Some(b"FCSTRIP1".as_slice()) || bytes.len() < 16 {
        return Err(invalid("not a stream strip file"));
    }
    let word = |at: usize| u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap()) as usize;
    let (width, rows) = (word(8), word(12));
    if width < 8 || rows == 0 {
        return Err(invalid("invalid strip size"));
    }
    let eye = width * rows;
    let size = 24 + 2 * eye;
    let frames = bytes[16..]
        .chunks_exact(size)
        .map(|record| {
            let report = u64::from_le_bytes(record[..8].try_into().unwrap());
            let strip = |index: usize| {
                record[24 + index * eye..24 + (index + 1) * eye]
                    .iter()
                    .map(|&value| f64::from(value))
                    .collect()
            };
            (report, [strip(0), strip(1)])
        })
        .collect();
    Ok(Strips {
        width,
        rows,
        frames,
    })
}

/// Pose times in nanoseconds with unwrapped head yaw in degrees, and the
/// eye tangents when recorded.
/// Pose times (ns), unwrapped yaw (degrees) and eye tangents.
type Poses = (Vec<f64>, Vec<f64>, [[f64; 4]; 2]);

fn read_poses(reader: impl BufRead) -> io::Result<Poses> {
    let (mut times, mut yaws) = (Vec::new(), Vec::<f64>::new());
    let mut tangents = FRAME_TANGENTS;
    for line in reader.lines() {
        let Ok(value) = serde_json::from_str::<Value>(&line?) else {
            continue;
        };
        if let Some(eyes) = value["eyeTangents"].as_array() {
            for (eye, values) in eyes.iter().take(2).enumerate() {
                for (index, tangent) in values.as_array().into_iter().flatten().take(4).enumerate()
                {
                    tangents[eye][index] =
                        tangent.as_f64().ok_or_else(|| invalid("bad tangent"))?;
                }
            }
        }
        let (Some(time), Some(matrix)) = (value["pose"].as_f64(), value["m"].as_array()) else {
            continue;
        };
        let column = |row: usize| {
            matrix
                .get(row)
                .and_then(|row| row.get(2))
                .and_then(Value::as_f64)
                .ok_or_else(|| invalid("bad pose matrix"))
        };
        // Head forward is -Z; yaw about standing space's vertical axis.
        let (x, z) = (-column(0)?, -column(2)?);
        let mut yaw = x.atan2(-z).to_degrees();
        if let Some(&previous) = yaws.last() {
            yaw += 360.0 * ((previous - yaw) / 360.0).round();
        }
        times.push(time);
        yaws.push(yaw);
    }
    if times.len() < 2 {
        return Err(invalid("fewer than two poses"));
    }
    Ok((times, yaws, tangents))
}

fn interpolate(times: &[f64], values: &[f64], at: f64) -> f64 {
    let index = times.partition_point(|&time| time < at);
    if index == 0 {
        return values[0];
    }
    if index == times.len() {
        return values[values.len() - 1];
    }
    let (t0, t1) = (times[index - 1], times[index]);
    let fraction = if t1 > t0 { (at - t0) / (t1 - t0) } else { 0.0 };
    values[index - 1] + fraction * (values[index] - values[index - 1])
}

/// Degrees per strip pixel. The strip covers the middle half of the eye's
/// width, which the probe samples.
fn degrees_per_pixel(tangents: [f64; 4], width: usize) -> f64 {
    let [left, right, ..] = tangents;
    let at = |u: f64| (left + u * (right - left)).atan().to_degrees();
    (at(0.75) - at(0.25)) / width as f64
}

/// Sideways shift of `second` relative to `first` in pixels (positive when
/// content moved right), from normalised cross-correlation of horizontal
/// gradients, with the peak correlation.
fn shift(first: &[f64], second: &[f64], width: usize, rows: usize) -> (f64, f64) {
    let gradients = |strip: &[f64]| {
        let mut out = vec![0.0; rows * (width - 1)];
        for row in 0..rows {
            let line = &strip[row * width..(row + 1) * width];
            let mean = (line[width - 1] - line[0]) / (width - 1) as f64;
            for x in 0..width - 1 {
                out[row * (width - 1) + x] = line[x + 1] - line[x] - mean;
            }
        }
        out
    };
    let (a, b) = (gradients(first), gradients(second));
    let w = width as i32 - 1;
    let limit = SHIFT_LIMIT.min(w - 8);
    let score = |d: i32| {
        let (mut ab, mut aa, mut bb) = (0.0, 0.0, 0.0);
        for row in 0..rows {
            let base = row * (w as usize);
            for x in 0.max(-d)..w.min(w - d) {
                let (p, q) = (a[base + x as usize], b[base + (x + d) as usize]);
                ab += p * q;
                aa += p * p;
                bb += q * q;
            }
        }
        if aa > 0.0 && bb > 0.0 {
            ab / (aa * bb).sqrt()
        } else {
            -1.0
        }
    };
    let scores: Vec<f64> = (-limit..=limit).map(score).collect();
    let (best, &peak) = scores
        .iter()
        .enumerate()
        .max_by(|x, y| x.1.total_cmp(y.1))
        .expect("non-empty search");
    let mut offset = best as f64 - f64::from(limit);
    if best > 0 && best + 1 < scores.len() {
        let (y0, y1, y2) = (scores[best - 1], peak, scores[best + 1]);
        let curvature = y0 - 2.0 * y1 + y2;
        if curvature < 0.0 {
            offset += 0.5 * (y0 - y2) / curvature;
        }
    }
    (offset, peak)
}

/// Report-time pairs and their measured picture rotation in degrees.
fn measure(strips: &Strips, tangents: [[f64; 4]; 2], gap_ns: f64) -> Vec<(f64, f64, f64)> {
    let reports: Vec<f64> = strips.frames.iter().map(|frame| frame.0 as f64).collect();
    let scales = tangents.map(|eye| degrees_per_pixel(eye, strips.width));
    let mut pairs = Vec::new();
    for k in 0..reports.len() {
        let j = reports.partition_point(|&time| time < reports[k] - gap_ns);
        if j >= k || reports[k] - reports[j] > 1.6 * gap_ns {
            continue;
        }
        let shifts: Vec<f64> = (0..2)
            .filter_map(|eye| {
                let (pixels, peak) = shift(
                    &strips.frames[j].1[eye],
                    &strips.frames[k].1[eye],
                    strips.width,
                    strips.rows,
                );
                (peak > MIN_PEAK).then_some(pixels * scales[eye])
            })
            .collect();
        if let [left, right] = shifts[..]
            && (left - right).abs() < MAX_EYE_DISAGREEMENT_DEG
        {
            pairs.push((reports[j], reports[k], 0.5 * (left + right)));
        }
    }
    pairs
}

/// Pearson correlation and least-squares slope of measured on predicted.
fn fit(predicted: &[f64], measured: &[f64]) -> (f64, f64) {
    let n = predicted.len() as f64;
    let (mx, my) = (
        predicted.iter().sum::<f64>() / n,
        measured.iter().sum::<f64>() / n,
    );
    let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
    for (x, y) in predicted.iter().zip(measured) {
        sxy += (x - mx) * (y - my);
        sxx += (x - mx) * (x - mx);
        syy += (y - my) * (y - my);
    }
    if sxx <= 0.0 || syy <= 0.0 {
        return (0.0, 0.0);
    }
    (sxy / (sxx * syy).sqrt(), sxy / sxx)
}

/// The delay in milliseconds with the strongest (negative) correlation, its
/// correlation and slope. The picture moves against the head, so a correct
/// delay gives a strongly negative correlation.
fn best_delay(
    pairs: &[(f64, f64, f64)],
    pose: (&[f64], &[f64]),
    delays: impl Iterator<Item = i32>,
) -> (i32, f64, f64) {
    let measured: Vec<f64> = pairs.iter().map(|pair| pair.2).collect();
    let mut best = (0, f64::INFINITY, 0.0);
    for delay in delays {
        let d = f64::from(delay) * 1e6;
        let predicted: Vec<f64> = pairs
            .iter()
            .map(|&(from, to, _)| {
                interpolate(pose.0, pose.1, to - d) - interpolate(pose.0, pose.1, from - d)
            })
            .collect();
        let (r, slope) = fit(&predicted, &measured);
        if r < best.1 {
            best = (delay, r, slope);
        }
    }
    best
}

fn main() -> Result<()> {
    let options = Options::parse();
    let strips = read_strips(&fs::read(&options.strips)?)?;
    let (times, yaws, tangents) = read_poses(BufReader::new(fs::File::open(&options.poses)?))?;
    let pairs = measure(&strips, tangents, options.gap_ms * 1e6);
    if pairs.len() < 50 {
        return Err(invalid(format!(
            "only {} matched pairs; turn the head slowly",
            pairs.len()
        ))
        .into());
    }
    let (delay, r, slope) = best_delay(&pairs, (&times, &yaws), -120..=200);
    // Resample pairs to estimate the delay's spread.
    let mut seed = 0x9e37_79b9_7f4a_7c15_u64;
    let mut next = |bound: usize| {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed % bound as u64) as usize
    };
    let mut samples: Vec<i32> = (0..200)
        .map(|_| {
            let resampled: Vec<_> = (0..pairs.len()).map(|_| pairs[next(pairs.len())]).collect();
            best_delay(&resampled, (&times, &yaws), (-120..=200).step_by(2)).0
        })
        .collect();
    samples.sort_unstable();
    let rates: Vec<f64> = times
        .windows(2)
        .zip(yaws.windows(2))
        .filter(|(t, _)| t[1] > t[0])
        .map(|(t, y)| ((y[1] - y[0]) / ((t[1] - t[0]) / 1e9)).abs())
        .collect();
    let mut sorted = rates.clone();
    sorted.sort_by(f64::total_cmp);
    let median_speed = sorted.get(sorted.len() / 2).copied().unwrap_or(0.0);
    // The picture should follow the head almost one to one. Fast turns
    // leave only matches on things that did not move, which shrinks the
    // slope and moves the delay.
    let reliable = r < -0.8 && (-1.2..=-0.6).contains(&slope) && median_speed <= 70.0;
    println!(
        "{:#}",
        json!({
            "delayMs": delay, "reliable": reliable, "correlation": r, "slope": slope,
            "delayMs5": samples[samples.len() / 20], "delayMs95": samples[samples.len() * 19 / 20],
            "pairs": pairs.len(), "frames": strips.frames.len(),
            "headSpeedDegPerSecMedian": median_speed,
            "note": if reliable {
                "set restFrameStreamPoseOffsetMs to delayMs"
            } else {
                "unreliable: record again turning the head slowly (20 to 60 deg/s), head only, facing detailed scenery"
            },
        })
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Strips of a random panorama seen through the Frame's projection with
    /// the head yawing, shown `delay_ms` late.
    fn synthetic(delay_ms: f64) -> (Strips, Vec<f64>, Vec<f64>) {
        let (width, rows) = (256, 16);
        let mut seed = 12345_u64;
        let mut noise = || {
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            ((seed >> 33) as f64 / f64::from(1_u32 << 31)) - 0.5
        };
        let panorama: Vec<f64> = {
            let raw: Vec<f64> = (0..rows * 3600).map(|_| noise()).collect();
            (0..rows * 3600)
                .map(|i| {
                    let (row, column) = (i / 3600, i % 3600);
                    // A 2.5° box blur keeps detail above the strip's
                    // pixel spacing, as in a real scene.
                    (0..25)
                        .map(|k| raw[row * 3600 + (column + k) % 3600])
                        .sum::<f64>()
                        * 40.0
                        + 120.0
                })
                .collect()
        };
        // A slow scan like the recorded calibration: ±15° every 2.5 s,
        // peaking at 38°/s.
        let yaw = |t: f64| 15.0 * (t / 2.5e9 * std::f64::consts::TAU).sin();
        let times: Vec<f64> = (0..5000).map(|i| f64::from(i) * 4e6).collect();
        let yaws: Vec<f64> = times.iter().map(|&t| yaw(t)).collect();
        let frames = (0..1300)
            .map(|i| {
                let report = 0.5e9 + f64::from(i) * 1e9 / 72.0;
                let content = yaw(report - delay_ms * 1e6);
                let strip = |eye: usize| {
                    let scale = FRAME_TANGENTS[eye];
                    let mut strip = vec![0.0; width * rows];
                    for x in 0..width {
                        let u = 0.25 + (x as f64 + 0.5) / width as f64 * 0.5;
                        let angle = (scale[0] + u * (scale[1] - scale[0])).atan().to_degrees();
                        let column =
                            (((angle + content) * 10.0).round() as i64).rem_euclid(3600) as usize;
                        for row in 0..rows {
                            strip[row * width + x] = panorama[row * 3600 + column];
                        }
                    }
                    strip
                };
                (report as u64, [strip(0), strip(1)])
            })
            .collect();
        (
            Strips {
                width,
                rows,
                frames,
            },
            times,
            yaws,
        )
    }

    #[test]
    fn recovers_a_known_stream_delay() {
        for delay in [0.0, 30.0, -20.0] {
            let (strips, times, yaws) = synthetic(delay);
            let pairs = measure(&strips, FRAME_TANGENTS, 50e6);
            assert!(pairs.len() > 500, "{} pairs", pairs.len());
            let (found, r, slope) = best_delay(&pairs, (&times, &yaws), -60..=80);
            assert!(
                (f64::from(found) - delay).abs() <= 2.0,
                "{delay}: found {found}"
            );
            assert!(r < -0.95 && (-1.1..-0.8).contains(&slope), "{r} {slope}");
        }
    }

    #[test]
    fn reads_the_probe_formats() {
        let mut bytes = b"FCSTRIP1".to_vec();
        bytes.extend(8_u32.to_le_bytes());
        bytes.extend(1_u32.to_le_bytes());
        bytes.extend(7_u64.to_le_bytes());
        bytes.extend([0_u8; 16]);
        bytes.extend((0..16).map(|value| value as u8));
        let strips = read_strips(&bytes).unwrap();
        assert_eq!((strips.width, strips.rows, strips.frames.len()), (8, 1, 1));
        assert_eq!(strips.frames[0].0, 7);
        assert_eq!(strips.frames[0].1[1][0], 8.0);
        let poses = "{\"eyeTangents\":[[-1,1,-1,1],[-2,2,-2,2]]}\n\
            {\"pose\":0,\"m\":[[1,0,0,0],[0,1,0,0],[0,0,1,0]]}\n\
            {\"pose\":1000,\"m\":[[0,0,-1,0],[0,1,0,0],[1,0,0,0]]}\n";
        let (times, yaws, tangents) = read_poses(poses.as_bytes()).unwrap();
        assert_eq!(times, [0.0, 1000.0]);
        assert_eq!(tangents[1][0], -2.0);
        assert!(
            yaws[0].abs() < 1e-9 && (yaws[1] - 90.0).abs() < 1e-9,
            "{yaws:?}"
        );
    }
}
