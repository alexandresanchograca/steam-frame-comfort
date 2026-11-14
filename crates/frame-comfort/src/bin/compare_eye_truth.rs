//! Compare labelled PC mirror grids with the Frame's captured stream rows.
use clap::Parser;
use serde_json::{Value, json};
use std::{
    error::Error,
    fs::File,
    io::{self, BufRead, BufReader},
    path::PathBuf,
};

const SIDE: usize = 32;
type Grid = [f64; SIDE * SIDE];

#[derive(Parser)]
#[command(about = "Compare labelled PC mirror grids with Frame Comfort --probe-grid output")]
struct Options {
    #[arg(required_unless_present = "self_test")]
    pc_json: Option<PathBuf>,
    #[arg(required_unless_present = "self_test")]
    frame_json: Option<PathBuf>,
    #[arg(long, conflicts_with_all = ["pc_json", "frame_json"])]
    self_test: bool,
}

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

fn grid(value: &Value, label: &str) -> io::Result<Grid> {
    let values = value
        .as_array()
        .filter(|values| values.len() == SIDE * SIDE)
        .ok_or_else(|| invalid(format!("{label} must be a 32×32 luma grid")))?;
    let mut result = [0.0; SIDE * SIDE];
    for (out, value) in result.iter_mut().zip(values) {
        *out = value
            .as_f64()
            .filter(|v| (0.0..=255.0).contains(v))
            .ok_or_else(|| invalid(format!("{label} contains invalid luma")))?;
    }
    Ok(result)
}

fn pc_grids(pc: &Value) -> io::Result<[Grid; 2]> {
    let eyes = pc["eyes"]
        .as_array()
        .ok_or_else(|| invalid("PC capture has no eyes array"))?;
    let find_eye = |label: &str| {
        let value = eyes
            .iter()
            .find(|eye| eye["eye"] == label && eye.get("grid").is_some())
            .ok_or_else(|| invalid(format!("PC capture has no {label} eye grid")))?;
        grid(&value["grid"], label)
    };
    Ok([find_eye("left")?, find_eye("right")?])
}

fn frame_grids(reader: impl BufRead) -> Result<[Grid; 2], Box<dyn Error>> {
    for line in reader.lines() {
        let line = line?;
        if line.contains("\"quadrants\"") {
            let frame: Value = serde_json::from_str(&line)?;
            let quadrants = &frame["quadrants"];
            return Ok([
                grid(&quadrants["top"], "top")?,
                grid(&quadrants["bottom"], "bottom")?,
            ]);
        }
    }
    Err(invalid("Frame capture contains no quadrant grids").into())
}

fn crop(grid: &Grid, scale: f64, shift: f64) -> Grid {
    let origin = 0.5 * SIDE as f64 * (1.0 - scale);
    let axis = |pixel: usize, shift: f64| {
        let sample = origin + shift + (pixel as f64 + 0.5) * scale - 0.5;
        let low = sample.floor().clamp(0.0, (SIDE - 1) as f64) as usize;
        let high = (low + 1).min(SIDE - 1);
        (low, high, (sample - low as f64).clamp(0.0, 1.0))
    };
    std::array::from_fn(|index| {
        let (x0, x1, dx) = axis(index % SIDE, shift);
        let (y0, y1, dy) = axis(index / SIDE, 0.0);
        let top = grid[y0 * SIDE + x0] * (1.0 - dx) + grid[y0 * SIDE + x1] * dx;
        let bottom = grid[y1 * SIDE + x0] * (1.0 - dx) + grid[y1 * SIDE + x1] * dx;
        top * (1.0 - dy) + bottom * dy
    })
}

fn correlation(a: &Grid, b: &Grid) -> f64 {
    let mean = |grid: &Grid| grid.iter().sum::<f64>() / grid.len() as f64;
    let (a_mean, b_mean) = (mean(a), mean(b));
    let (mut covariance, mut a_variance, mut b_variance) = (0.0, 0.0, 0.0);
    for (&a, &b) in a.iter().zip(b) {
        let (a, b) = (a - a_mean, b - b_mean);
        covariance += a * b;
        a_variance += a * a;
        b_variance += b * b;
    }
    if a_variance == 0.0 || b_variance == 0.0 {
        return 0.0;
    }
    covariance / (a_variance.sqrt() * b_variance.sqrt())
}

fn best_match(eye: &Grid, row: &Grid) -> f64 {
    let mut best = -1.0_f64;
    // The PC mirror is narrower than the stream's full eye view.
    for scale in (60..=100).step_by(4) {
        for shift in -8..=8 {
            let crop = crop(row, f64::from(scale) / 100.0, f64::from(shift) / 2.0);
            best = best.max(correlation(eye, &crop));
        }
    }
    best
}

fn decide([left, right]: &[Grid; 2], [top, bottom]: &[Grid; 2]) -> Value {
    let [left_top, left_bottom, right_top, right_bottom] =
        [(left, top), (left, bottom), (right, top), (right, bottom)]
            .map(|(eye, row)| best_match(eye, row));
    let straight = left_top + right_bottom;
    let crossed = left_bottom + right_top;
    let verdict = if straight - crossed > 0.05 {
        "top=left, bottom=right"
    } else if crossed - straight > 0.05 {
        "top=right, bottom=left"
    } else {
        "inconclusive"
    };
    let rounded = |score: f64| (score * 1_000.0).round_ties_even() / 1_000.0;
    json!({
        "scores": {
            "left-top": rounded(left_top),
            "left-bottom": rounded(left_bottom),
            "right-top": rounded(right_top),
            "right-bottom": rounded(right_bottom),
        },
        "straight": rounded(straight),
        "crossed": rounded(crossed),
        "verdict": verdict,
    })
}

fn synthetic(offset: usize) -> Grid {
    let mut grid = [20.0; SIDE * SIDE];
    for row in grid.chunks_exact_mut(SIDE).skip(8).take(8) {
        row[offset..offset + 12].fill(200.0);
    }
    grid[20 * SIDE + 5] = 255.0;
    grid
}

fn self_test() -> io::Result<()> {
    let pc = [synthetic(14), synthetic(6)];
    for (rows, expected) in [
        (pc, "top=left, bottom=right"),
        ([pc[1], pc[0]], "top=right, bottom=left"),
        ([synthetic(10); 2], "inconclusive"),
    ] {
        let comparison = decide(&pc, &rows);
        if comparison["verdict"] != expected {
            return Err(invalid(format!("expected {expected}, got {comparison}")));
        }
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let options = Options::parse();
    if options.self_test {
        self_test()?;
        println!("compare_eye_truth self-test passed");
        return Ok(());
    }
    let pc: Value = serde_json::from_reader(BufReader::new(File::open(options.pc_json.unwrap())?))?;
    let frame = frame_grids(BufReader::new(File::open(options.frame_json.unwrap())?))?;
    println!("{}", decide(&pc_grids(&pc)?, &frame));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eye_order_matches_the_baseline_self_test() {
        self_test().unwrap();
    }

    #[test]
    fn inputs_use_eye_labels_and_the_first_reported_quadrants() {
        let left = synthetic(14);
        let right = synthetic(6);
        let pc = json!({"eyes": [
            {"eye": "right", "grid": right.as_slice()},
            {"eye": "left", "grid": left.as_slice()},
        ]});
        let frame = json!({"quadrants": {"top": right.as_slice(), "bottom": left.as_slice()}});
        let log = format!("startup noise\n{frame}\n{{\"quadrants\":{{}}}}\n");
        assert_eq!(pc_grids(&pc).unwrap(), [left, right]);
        assert_eq!(frame_grids(log.as_bytes()).unwrap(), [right, left]);
        assert!(frame_grids(b"no frames\n".as_slice()).is_err());
        assert!(grid(&json!([0, 255]), "short grid").is_err());
    }
}
