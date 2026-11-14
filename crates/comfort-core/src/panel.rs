//! Pixel-based SteamVR dashboard layout, shared by drawing and hit testing.
use crate::{
    settings::{OpeningShape, Periphery, Settings, Stick},
    timing::opening_degrees,
};

pub const WIDTH: usize = 1024;
pub const HEIGHT: usize = 640;

pub type Pixel = [u8; 4];

pub struct Image {
    pub width: usize,
    pub height: usize,
    pub pixels: Vec<Pixel>,
}

#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub enum Action {
    #[default]
    None,
    SaveSettings,
    Quit,
    /// Restart SteamVR so its compositor loads the installed observer.
    RestartSteamVr,
    /// Download and install the newer release.
    Update,
}

/// Installation state shown in the panel's footer, with a button where the
/// wearer can act on it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum System {
    #[default]
    Ready,
    /// SteamVR's compositor has not loaded the observer, so Game mode cannot
    /// see the game.
    ObserverMissing,
    /// The compositor still runs the observer from before an update.
    ObserverOutdated,
    UpdateAvailable {
        version: String,
        failed: bool,
    },
    /// A restart or update is running; no button.
    Busy(&'static str),
}

impl System {
    fn restarts(&self) -> bool {
        matches!(self, Self::ObserverMissing | Self::ObserverOutdated)
    }
}

#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub struct Click {
    pub action: Action,
    pub changed: bool,
}

#[derive(Clone, Copy)]
struct Rect(i32, i32, i32, i32);

impl Rect {
    fn contains(self, [x, y]: [f32; 2]) -> bool {
        x >= self.0 as f32
            && x < (self.0 + self.2) as f32
            && y >= self.1 as f32
            && y < (self.1 + self.3) as f32
    }
}

#[derive(Clone, Copy, Default, PartialEq, Eq)]
enum Tab {
    #[default]
    Opening,
    Timing,
    Extras,
}

const ENABLED: Rect = Rect(862, 27, 138, 32);
const QUIT: Rect = Rect(820, 599, 180, 32);
const SYSTEM_BUTTON: Rect = Rect(600, 599, 210, 32);
const TABS: [Choice<Tab>; 3] = [
    Choice(Rect(24, 82, 154, 38), "OPENING", Tab::Opening),
    Choice(Rect(184, 82, 154, 38), "TIMING", Tab::Timing),
    Choice(Rect(344, 82, 154, 38), "EXTRAS", Tab::Extras),
];

struct Choice<T>(Rect, &'static str, T);
const MODES: [Choice<Periphery>; 3] = [
    Choice(Rect(145, 159, 98, 32), "PLAIN", Periphery::Plain),
    Choice(Rect(252, 159, 98, 32), "GRID", Periphery::Grid),
    Choice(Rect(359, 159, 108, 32), "GAME", Periphery::Game),
];
const STICKS: [Choice<Stick>; 3] = [
    Choice(Rect(145, 202, 98, 32), "LEFT", Stick::Left),
    Choice(Rect(252, 202, 98, 32), "RIGHT", Stick::Right),
    Choice(Rect(359, 202, 108, 32), "EITHER", Stick::Both),
];
const SHAPES: [Choice<OpeningShape>; 2] = [
    Choice(Rect(145, 245, 135, 32), "CIRCLE", OpeningShape::Circle),
    Choice(
        Rect(290, 245, 177, 32),
        "RECTANGLE",
        OpeningShape::Rectangle,
    ),
];

fn choose<T: Copy>(point: [f32; 2], choices: &[Choice<T>], value: &mut T) {
    if let Some(Choice(_, _, selected)) = choices.iter().find(|choice| choice.0.contains(point)) {
        *value = *selected;
    }
}

struct Toggle {
    bounds: Rect,
    label: &'static str,
    field: fn(&mut Settings) -> &mut bool,
}
const DYNAMIC: Toggle = Toggle {
    bounds: Rect(359, 332, 108, 32),
    label: "DYNAMIC FOV",
    field: |s| &mut s.dynamic_fov,
};
const EXTRAS: [Toggle; 3] = [
    Toggle {
        bounds: Rect(862, 190, 118, 34),
        label: "GRID OVER RING",
        field: |s| &mut s.grid_over_ring,
    },
    Toggle {
        bounds: Rect(862, 276, 118, 34),
        label: "FAKE NOSE",
        field: |s| &mut s.nose,
    },
    Toggle {
        bounds: Rect(862, 362, 118, 34),
        label: "CROSSHAIR",
        field: |s| &mut s.crosshair,
    },
];

#[derive(Clone, Copy)]
enum Unit {
    Degrees,
    Percent,
    Meters,
    Centimeters,
    Milliseconds,
    MillisecondsOrOff,
    Seconds,
}

impl Unit {
    fn format(self, value: f32) -> String {
        match self {
            Self::Degrees => format!("{} DEG", value.round() as i32),
            Self::Percent => format!("{}%", (value * 100.0).round() as i32),
            Self::Meters => format!("{value:.2} M"),
            Self::Centimeters => format!("{value:.1} CM"),
            Self::MillisecondsOrOff if value <= 0.0 => "OFF".into(),
            Self::Milliseconds | Self::MillisecondsOrOff => format!("{} MS", value.round() as i32),
            Self::Seconds => format!("{:.2} S", value / 1000.0),
        }
    }
}

struct Slider {
    track: Rect,
    label: &'static str,
    field: fn(&mut Settings) -> &mut f32,
    range: [f32; 2],
    step: f32,
    unit: Unit,
    min_width: bool,
}

impl Slider {
    fn enabled(&self, settings: &Settings) -> bool {
        !self.min_width || settings.dynamic_fov
    }

    fn bounds(&self, settings: &Settings) -> [f32; 2] {
        if self.min_width {
            [self.range[0], settings.opening_width_deg]
        } else {
            self.range
        }
    }
}

macro_rules! slider {
    ($x:literal, $y:literal, $label:literal, $field:ident, $min:literal, $max:literal, $step:literal, $unit:ident, $min_width:literal) => {
        Slider {
            track: Rect($x, $y, 211, 30),
            label: $label,
            field: |s| &mut s.$field,
            range: [$min, $max],
            step: $step,
            unit: Unit::$unit,
            min_width: $min_width,
        }
    };
}

const OPENING_SLIDERS: [Slider; 5] = [
    slider!(
        180,
        293,
        "WIDTH",
        opening_width_deg,
        20.0,
        140.0,
        5.0,
        Degrees,
        false
    ),
    slider!(
        180,
        373,
        "MIN WIDTH",
        opening_min_width_deg,
        20.0,
        140.0,
        5.0,
        Degrees,
        true
    ),
    slider!(
        180,
        413,
        "HEIGHT",
        opening_height_deg,
        20.0,
        120.0,
        5.0,
        Degrees,
        false
    ),
    slider!(
        180,
        453,
        "EDGE",
        opening_fade,
        0.0,
        0.6,
        0.05,
        Percent,
        false
    ),
    slider!(
        180,
        493,
        "OPACITY",
        periphery_opacity,
        0.2,
        1.0,
        0.05,
        Percent,
        false
    ),
];
const TIMING_SLIDERS: [Slider; 6] = [
    slider!(
        180,
        190,
        "FADE IN",
        fade_in_ms,
        0.0,
        1000.0,
        50.0,
        Milliseconds,
        false
    ),
    slider!(
        180,
        276,
        "DELAY",
        fade_out_delay_ms,
        0.0,
        3000.0,
        100.0,
        Milliseconds,
        false
    ),
    slider!(
        180,
        362,
        "FADE OUT",
        fade_out_ms,
        0.0,
        3000.0,
        100.0,
        Milliseconds,
        false
    ),
    slider!(
        674,
        190,
        "REFRESH",
        rest_frame_interval_ms,
        500.0,
        3000.0,
        250.0,
        Seconds,
        false
    ),
    slider!(
        674,
        276,
        "BLINK",
        refresh_blink_ms,
        0.0,
        400.0,
        20.0,
        MillisecondsOrOff,
        false
    ),
    slider!(
        674,
        362,
        "DIM",
        periphery_dim,
        0.0,
        0.9,
        0.05,
        Percent,
        false
    ),
];
const EXTRA_SLIDERS: [Slider; 4] = [
    slider!(
        180, 190, "SHADE", brightness, 0.03, 0.4, 0.01, Percent, false
    ),
    slider!(
        180,
        248,
        "LINES",
        grid_brightness,
        0.03,
        0.6,
        0.01,
        Percent,
        false
    ),
    slider!(
        180,
        306,
        "THICKNESS",
        grid_thickness_cm,
        0.5,
        10.0,
        0.5,
        Centimeters,
        false
    ),
    slider!(
        180,
        364,
        "SPACING",
        grid_spacing,
        0.25,
        4.0,
        0.25,
        Meters,
        false
    ),
];

const BG: Pixel = [16, 22, 32, 255];
const SURFACE: Pixel = [29, 41, 56, 255];
const CONTROL: Pixel = [38, 53, 69, 255];
const LINE: Pixel = [68, 87, 105, 255];
const INK: Pixel = [239, 244, 249, 255];
const MUTED: Pixel = [180, 197, 210, 255];
const ACCENT: Pixel = [99, 229, 203, 255];
const WARNING: Pixel = [255, 190, 173, 255];
const EXIT: Pixel = [77, 43, 49, 255];

struct Canvas(Image);

impl Canvas {
    fn new(width: usize, height: usize) -> Self {
        Self(Image {
            width,
            height,
            pixels: vec![BG; width * height],
        })
    }

    fn fill(&mut self, Rect(x, y, width, height): Rect, color: Pixel) {
        let left = x.max(0);
        let right = (x + width).min(self.0.width as i32);
        if left < right {
            for row in y.max(0)..(y + height).min(self.0.height as i32) {
                let offset = row as usize * self.0.width;
                self.0.pixels[offset + left as usize..offset + right as usize].fill(color);
            }
        }
    }

    fn text(&mut self, mut x: i32, y: i32, text: &str, scale: i32, color: Pixel) {
        for letter in text.bytes() {
            if let Some(glyph) = GLYPH_NAMES.iter().position(|&ch| ch == letter) {
                for (row, bits) in GLYPHS[glyph].into_iter().enumerate() {
                    for col in 0..5 {
                        if bits & (16 >> col) != 0 {
                            self.fill(
                                Rect(x + col * scale, y + row as i32 * scale, scale, scale),
                                color,
                            );
                        }
                    }
                }
            }
            x += 6 * scale;
        }
    }

    fn button(&mut self, bounds: Rect, label: &str, base: Pixel, fg: Pixel) {
        self.fill(bounds, base);
        let width = (label.len() as i32 * 6 - 1) * 2;
        self.text(
            bounds.0 + (bounds.2 - width) / 2,
            bounds.1 + (bounds.3 - 14) / 2,
            label,
            2,
            fg,
        );
    }

    fn choice(&mut self, bounds: Rect, label: &str, selected: bool) {
        let (base, fg) = if selected {
            (ACCENT, BG)
        } else {
            (CONTROL, INK)
        };
        self.button(bounds, label, base, fg);
    }

    fn choices<T: PartialEq>(&mut self, label: &str, choices: &[Choice<T>], selected: T) {
        self.text(44, choices[0].0.1 + 9, label, 2, INK);
        for Choice(bounds, text, value) in choices {
            self.choice(*bounds, text, selected == *value);
        }
    }

    fn toggle(&mut self, x: i32, toggle: &Toggle, settings: &mut Settings) {
        let on = *(toggle.field)(settings);
        self.text(
            x,
            toggle.bounds.1 + (toggle.bounds.3 - 14) / 2,
            toggle.label,
            2,
            INK,
        );
        self.choice(toggle.bounds, if on { "ON" } else { "OFF" }, on);
    }

    fn slider(&mut self, x: i32, slider: &Slider, settings: &mut Settings) {
        let Rect(left, top, width, height) = slider.track;
        let value = *(slider.field)(settings);
        let [min, max] = slider.bounds(settings);
        let fraction = if max > min {
            ((value - min) / (max - min)).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let pos = (fraction * (width - 1) as f32).round() as i32;
        let y = top + height / 2;
        let (text, bar) = if slider.enabled(settings) {
            (INK, ACCENT)
        } else {
            (MUTED, MUTED)
        };
        self.text(x, y - 7, slider.label, 2, text);
        self.fill(Rect(left, y - 2, width, 4), LINE);
        self.fill(Rect(left, y - 2, pos + 1, 4), bar);
        self.fill(Rect(left + pos - 3, y - 7, 7, 14), bar);
        self.text(
            left + width + 10,
            y - 7,
            &slider.unit.format(value),
            2,
            text,
        );
    }

    fn preview(&mut self, Rect(x, y, width, height): Rect, settings: &Settings) {
        let side = width.min(height) - 2;
        let x0 = x + (width - side) / 2;
        let y0 = y + (height - side) / 2;
        self.fill(Rect(x0 - 1, y0 - 1, side + 2, side + 2), LINE);
        let tan = |angle: f32| (f64::from(angle).to_radians() * 0.5).tan();
        let full = [
            tan(settings.opening_width_deg),
            tan(settings.opening_height_deg),
        ];
        let minimum = opening_degrees(
            settings.opening_width_deg,
            settings.opening_height_deg,
            settings.opening_min_width_deg,
            1.0,
        )
        .map(tan);
        let tangent_step = 2.0 * tan(144.0) / f64::from(side);
        let outline = tangent_step / minimum[0].min(minimum[1]);
        let radius = |u: f64, v: f64, [w, h]: [f64; 2]| {
            let (x, y) = (u / w, v / h);
            if settings.shape == OpeningShape::Rectangle {
                x.abs().max(y.abs())
            } else {
                x.hypot(y)
            }
        };
        let coordinate = |n| (f64::from(n) + 0.5 - f64::from(side) * 0.5) * tangent_step;
        let shade = encoded_shade(settings.brightness);
        let line_shade = encoded_shade(settings.grid_brightness);
        let spacing = f64::from(settings.grid_spacing);
        let half_line = (f64::from(settings.grid_thickness_cm) * 0.005).max(tangent_step * 2.0);
        let grid = |t: f64| {
            let v = 4.0 * t;
            (v - (v / spacing).round_ties_even() * spacing).abs() < half_line
        };
        let feather = f64::from(settings.opening_fade);
        let strength = f64::from(settings.periphery_opacity);
        for y in 0..side {
            let v = coordinate(y);
            for x in 0..side {
                let u = coordinate(x);
                let index = (y0 + y) as usize * self.0.width + (x0 + x) as usize;
                if settings.dynamic_fov
                    && side >= 100
                    && (radius(u, v, minimum) - 1.0).abs() < outline
                {
                    self.0.pixels[index] = ACCENT;
                    continue;
                }
                let r = radius(u, v, full);
                let alpha = if feather < 1e-6 {
                    if r >= 1.0 { strength } else { 0.0 }
                } else {
                    let t = ((r - (1.0 - feather)) / feather).clamp(0.0, 1.0);
                    strength * t * t * (3.0 - 2.0 * t)
                };
                let mut still = match settings.periphery {
                    Periphery::Plain => [0; 3],
                    Periphery::Grid => [shade; 3],
                    Periphery::Game => {
                        let base = if (x + y) / 6 % 2 == 0 {
                            [54, 78, 96]
                        } else {
                            [44, 64, 80]
                        };
                        let keep = 1.0 - settings.periphery_dim.clamp(0.0, 1.0);
                        base.map(|value| (value as f32 * keep).round() as u8)
                    }
                };
                if (settings.periphery == Periphery::Grid || settings.grid_over_ring)
                    && (grid(u) || grid(v))
                {
                    still = [line_shade; 3];
                }
                let mix = |live: u8, still: u8| {
                    let live = f64::from(live);
                    (live + (f64::from(still) - live) * alpha).round() as u8
                };
                self.0.pixels[index] = [
                    mix(150, still[0]),
                    mix(178, still[1]),
                    mix(200, still[2]),
                    255,
                ];
            }
        }
        self.fill(Rect(x0 + side / 2 - 3, y0 + side / 2, 7, 1), BG);
        self.fill(Rect(x0 + side / 2, y0 + side / 2 - 3, 1, 7), BG);
    }
}

fn encoded_shade(linear: f32) -> u8 {
    let encoded = if linear <= 0.0031308 {
        12.92 * linear
    } else {
        1.055 * linear.powf(1.0 / 2.4) - 0.055
    };
    (255.0 * encoded).round() as u8
}

#[derive(Default)]
pub struct Panel {
    tab: Tab,
    save_failed: bool,
    /// The restart button was pressed once and now asks for confirmation.
    armed: bool,
}

impl Panel {
    pub fn click(&mut self, point: [f32; 2], settings: &mut Settings, system: &System) -> Click {
        let was_armed = std::mem::take(&mut self.armed);
        if SYSTEM_BUTTON.contains(point) {
            let action = match system {
                // Restarting SteamVR ends a stream, so it takes a second press.
                system if system.restarts() && !was_armed => {
                    self.armed = true;
                    Action::None
                }
                system if system.restarts() => Action::RestartSteamVr,
                System::UpdateAvailable { .. } => Action::Update,
                _ => Action::None,
            };
            return Click {
                action,
                changed: self.armed != was_armed || action != Action::None,
            };
        }
        let click = self.click_settings(point, settings);
        Click {
            changed: click.changed || was_armed,
            ..click
        }
    }

    fn click_settings(&mut self, point: [f32; 2], settings: &mut Settings) -> Click {
        if QUIT.contains(point) {
            return Click {
                action: Action::Quit,
                changed: false,
            };
        }
        if ENABLED.contains(point) {
            settings.enabled = !settings.enabled;
            return Click {
                action: Action::None,
                changed: true,
            };
        }
        if let Some(Choice(_, _, tab)) = TABS.iter().find(|choice| choice.0.contains(point)) {
            let changed = self.tab != *tab;
            self.tab = *tab;
            return Click {
                action: Action::None,
                changed,
            };
        }
        let before = settings.clone();
        let (sliders, toggles): (&[Slider], &[Toggle]) = match self.tab {
            Tab::Opening => {
                choose(point, &MODES, &mut settings.periphery);
                choose(point, &STICKS, &mut settings.stick);
                choose(point, &SHAPES, &mut settings.shape);
                (&OPENING_SLIDERS, std::slice::from_ref(&DYNAMIC))
            }
            Tab::Timing => (&TIMING_SLIDERS, &[]),
            Tab::Extras => (&EXTRA_SLIDERS, &EXTRAS),
        };
        for toggle in toggles {
            if toggle.bounds.contains(point) {
                let value = (toggle.field)(settings);
                *value = !*value;
            }
        }
        if let Some(slider) = sliders
            .iter()
            .find(|slider| slider.track.contains(point) && slider.enabled(settings))
        {
            let [min, max] = slider.bounds(settings);
            let fraction =
                ((point[0] - slider.track.0 as f32) / (slider.track.2 - 1) as f32).clamp(0.0, 1.0);
            let stepped = min + (fraction * (max - min) / slider.step).round() * slider.step;
            let value = if fraction == 1.0 {
                max
            } else {
                stepped.clamp(min, max)
            };
            let field = (slider.field)(settings);
            if (*field - value).abs() > 0.001 {
                *field = value;
            }
        }
        if *settings == before {
            return Click::default();
        }
        settings.clamp();
        if *settings == before {
            return Click::default();
        }
        self.save_failed = false;
        Click {
            action: Action::SaveSettings,
            changed: true,
        }
    }

    pub fn save_failed(&mut self) {
        self.save_failed = true;
    }

    pub fn render(&self, settings: &Settings, game_source: &str, system: &System) -> Image {
        let mut settings = settings.clone();
        let mut canvas = Canvas::new(WIDTH, HEIGHT);
        canvas.fill(Rect(16, 16, 992, 54), SURFACE);
        canvas.text(27, 32, "FRAME COMFORT", 3, INK);
        canvas.text(275, 39, crate::release::CURRENT, 2, MUTED);
        canvas.choice(
            ENABLED,
            if settings.enabled {
                "ENABLED"
            } else {
                "DISABLED"
            },
            settings.enabled,
        );
        for Choice(bounds, label, tab) in TABS {
            canvas.choice(bounds, label, self.tab == tab);
        }
        canvas.fill(Rect(24, 140, 464, 428), SURFACE);
        canvas.fill(Rect(504, 140, 496, 428), SURFACE);
        match self.tab {
            Tab::Opening => {
                canvas.choices("MODE", &MODES, settings.periphery);
                if settings.periphery == Periphery::Game {
                    canvas.text(145, 193, source_status(game_source), 1, MUTED);
                }
                canvas.choices("STICK", &STICKS, settings.stick);
                canvas.choices("SHAPE", &SHAPES, settings.shape);
                canvas.toggle(44, &DYNAMIC, &mut settings);
                for slider in &OPENING_SLIDERS {
                    canvas.slider(44, slider, &mut settings);
                }
                canvas.text(44, 539, "MIN WIDTH AT FULL STICK PUSH", 2, MUTED);
                canvas.text(524, 163, "OPENING PREVIEW", 2, INK);
                canvas.preview(Rect(524, 194, 456, 318), &settings);
                canvas.fill(Rect(524, 533, 12, 8), [150, 178, 200, 255]);
                canvas.text(544, 530, "OPENING AT REST", 2, MUTED);
                if settings.dynamic_fov {
                    canvas.fill(Rect(754, 533, 12, 8), ACCENT);
                    canvas.text(774, 530, "FULL STICK PUSH", 2, MUTED);
                }
            }
            Tab::Timing => {
                canvas.text(44, 163, "MOVEMENT FADES", 2, INK);
                canvas.text(524, 163, "GAME SNAPSHOTS", 2, INK);
                for slider in &TIMING_SLIDERS {
                    canvas.slider(
                        if slider.track.0 < 500 { 44 } else { 524 },
                        slider,
                        &mut settings,
                    );
                }
                for (x, y, text) in [
                    (44, 233, "TIME TO SHOW THE PERIPHERY"),
                    (44, 319, "TIME HELD AFTER STICK RELEASE"),
                    (44, 405, "TIME TO CLEAR THE PERIPHERY"),
                    (524, 233, "TIME BETWEEN GAME SNAPSHOTS"),
                    (524, 319, "DIP TO BLACK DURING A SNAPSHOT SWAP"),
                    (524, 405, "DARKEN THE FROZEN GAME PICTURE"),
                ] {
                    canvas.text(x, y, text, 2, MUTED);
                }
            }
            Tab::Extras => {
                canvas.text(44, 163, "GRID APPEARANCE", 2, INK);
                for slider in &EXTRA_SLIDERS {
                    canvas.slider(44, slider, &mut settings);
                }
                canvas.text(524, 163, "OVERLAYS", 2, INK);
                for toggle in &EXTRAS {
                    canvas.toggle(524, toggle, &mut settings);
                }
                canvas.text(524, 236, "ROOM-FIXED LINES OVER PLAIN OR GAME", 2, MUTED);
                canvas.text(524, 322, "VISIBLE LOW IN VIEW AT REST", 2, MUTED);
                canvas.text(524, 408, "VISIBLE STRAIGHT AHEAD AT REST", 2, MUTED);
            }
        }
        canvas.fill(Rect(24, 588, 976, 1), LINE);
        self.footer(&mut canvas, system);
        canvas.button(QUIT, "QUIT COMFORT", EXIT, WARNING);
        canvas.0
    }

    fn footer(&self, canvas: &mut Canvas, system: &System) {
        if self.save_failed {
            canvas.text(24, 610, "COULD NOT SAVE SETTINGS", 1, WARNING);
            return;
        }
        let version;
        let (message, button) = match system {
            System::Ready => {
                canvas.text(24, 610, "SETTINGS SAVE AUTOMATICALLY", 1, MUTED);
                return;
            }
            _ if self.armed && system.restarts() => (
                "THIS RESTARTS STEAMVR AND ENDS A STREAM",
                Some("CONFIRM RESTART"),
            ),
            System::ObserverMissing => (
                "GAME MODE NEEDS ONE STEAMVR RESTART",
                Some("RESTART STEAMVR"),
            ),
            System::ObserverOutdated => (
                "RESTART STEAMVR TO FINISH THE UPDATE",
                Some("RESTART STEAMVR"),
            ),
            System::UpdateAvailable { version: v, failed } => {
                version = if *failed {
                    format!("UPDATE TO {v} FAILED")
                } else {
                    format!("VERSION {v} IS AVAILABLE")
                };
                (
                    version.as_str(),
                    Some(if *failed { "TRY AGAIN" } else { "UPDATE" }),
                )
            }
            System::Busy(message) => (*message, None),
        };
        canvas.text(24, 608, message, 2, ACCENT);
        if let Some(label) = button {
            canvas.button(SYSTEM_BUTTON, label, ACCENT, BG);
        }
    }

    pub fn thumbnail() -> Image {
        let mut canvas = Canvas::new(48, 48);
        let settings = Settings {
            opening_width_deg: 90.0,
            opening_height_deg: 90.0,
            ..Settings::default()
        };
        canvas.preview(Rect(2, 4, 44, 40), &settings);
        canvas.0
    }
}

fn source_status(source: &str) -> &'static str {
    match source {
        "verified" => "GAME: SOURCE VERIFIED",
        "verified-next-movement" => "GAME: READY FOR NEXT MOVEMENT",
        "learning" => "GAME: NOT READY (LEARNING THIS GAME)",
        "no-fresh-source" | "no-stereo-source" => "GAME: NOT READY (WAITING FOR GAME)",
        "separate-eye-pairing-unverified" => "GAME: NOT READY (PAIRING THE TWO EYES)",
        "stereo-layout-unverified" => "GAME: NOT READY (CHECKING EYE LAYOUT)",
        "unsupported-stereo-layout" => "GAME: NOT READY (NO STEREO PICTURE FOUND)",
        "ambiguous-source" => "GAME: NOT READY (SEVERAL PICTURES FOUND)",
        "invalid-source-description" => "GAME: NOT READY (SOURCE ERROR)",
        "incomplete-stereo-report" => "GAME: NOT READY (INCOMPLETE SOURCE)",
        "waiting-for-complete-frame" => "GAME: NOT READY (WAITING FOR FRAME)",
        "projection-unavailable" => "GAME: NOT READY (WAITING FOR TRACKING)",
        _ => "GAME: NOT READY",
    }
}

// The original fixed-width font, so no system fonts are needed.
const GLYPH_NAMES: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789%-.():";
const GLYPHS: [[u8; 7]; 42] = [
    [14, 17, 17, 31, 17, 17, 17],
    [30, 17, 17, 30, 17, 17, 30],
    [14, 17, 16, 16, 16, 17, 14],
    [30, 17, 17, 17, 17, 17, 30],
    [31, 16, 16, 30, 16, 16, 31],
    [31, 16, 16, 30, 16, 16, 16],
    [14, 17, 16, 23, 17, 17, 14],
    [17, 17, 17, 31, 17, 17, 17],
    [14, 4, 4, 4, 4, 4, 14],
    [7, 2, 2, 2, 18, 18, 12],
    [17, 18, 20, 24, 20, 18, 17],
    [16, 16, 16, 16, 16, 16, 31],
    [17, 27, 21, 21, 17, 17, 17],
    [17, 25, 21, 19, 17, 17, 17],
    [14, 17, 17, 17, 17, 17, 14],
    [30, 17, 17, 30, 16, 16, 16],
    [14, 17, 17, 17, 21, 18, 13],
    [30, 17, 17, 30, 20, 18, 17],
    [15, 16, 16, 14, 1, 1, 30],
    [31, 4, 4, 4, 4, 4, 4],
    [17, 17, 17, 17, 17, 17, 14],
    [17, 17, 17, 17, 17, 10, 4],
    [17, 17, 17, 21, 21, 21, 10],
    [17, 17, 10, 4, 10, 17, 17],
    [17, 17, 10, 4, 4, 4, 4],
    [31, 1, 2, 4, 8, 16, 31],
    [14, 17, 19, 21, 25, 17, 14],
    [4, 12, 4, 4, 4, 4, 14],
    [14, 17, 1, 2, 4, 8, 31],
    [30, 1, 1, 14, 1, 1, 30],
    [2, 6, 10, 18, 31, 2, 2],
    [31, 16, 16, 30, 1, 1, 30],
    [14, 16, 16, 30, 17, 17, 14],
    [31, 1, 2, 4, 8, 8, 8],
    [14, 17, 17, 14, 17, 17, 14],
    [14, 17, 17, 15, 1, 1, 14],
    [17, 2, 4, 8, 16, 17, 0],
    [0, 0, 0, 31, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 4],
    [2, 4, 8, 8, 8, 4, 2],
    [8, 4, 2, 2, 2, 4, 8],
    [0, 4, 4, 0, 4, 4, 0],
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tabs_keep_the_existing_click_locations_and_save_rules() {
        let mut panel = Panel::default();
        let mut settings = Settings::default();
        assert_eq!(
            panel.click([920.0, 43.0], &mut settings, &System::Ready),
            Click {
                action: Action::None,
                changed: true
            }
        );
        assert!(!settings.enabled);
        assert_eq!(
            panel
                .click([301.0, 175.0], &mut settings, &System::Ready)
                .action,
            Action::SaveSettings
        );
        assert_eq!(settings.periphery, Periphery::Grid);
        assert_eq!(
            panel
                .click([390.0, 308.0], &mut settings, &System::Ready)
                .action,
            Action::SaveSettings
        );
        assert_eq!(settings.opening_width_deg, 140.0);
        assert_eq!(
            panel
                .click([261.0, 101.0], &mut settings, &System::Ready)
                .action,
            Action::None
        );
        assert_eq!(
            panel
                .click([884.0, 377.0], &mut settings, &System::Ready)
                .action,
            Action::SaveSettings
        );
        assert_eq!(settings.periphery_dim, 0.9);
        let image = panel.render(&settings, "verified", &System::Ready);
        assert_eq!(image.pixels.len(), WIDTH * HEIGHT);
        panel.save_failed();
        assert_ne!(
            panel.render(&settings, "verified", &System::Ready).pixels,
            image.pixels
        );
        panel.click([421.0, 101.0], &mut settings, &System::Ready);
        assert_eq!(
            panel
                .click([920.0, 293.0], &mut settings, &System::Ready)
                .action,
            Action::SaveSettings
        );
        assert!(settings.nose);
        assert_eq!(
            panel
                .click([900.0, 615.0], &mut settings, &System::Ready)
                .action,
            Action::Quit
        );
    }

    #[test]
    fn system_button_confirms_restarts_and_offers_updates() {
        let mut panel = Panel::default();
        let mut settings = Settings::default();
        let button = [705.0, 615.0];
        let ready = panel.render(&settings, "verified", &System::Ready);
        // No button while nothing needs doing.
        assert_eq!(
            panel.click(button, &mut settings, &System::Ready),
            Click::default()
        );
        let missing = System::ObserverMissing;
        let offer = panel.render(&settings, "verified", &missing);
        assert_ne!(offer.pixels, ready.pixels);
        // The first press asks for confirmation and redraws.
        assert_eq!(
            panel.click(button, &mut settings, &missing),
            Click {
                action: Action::None,
                changed: true
            }
        );
        assert_ne!(
            panel.render(&settings, "verified", &missing).pixels,
            offer.pixels
        );
        assert_eq!(
            panel.click(button, &mut settings, &missing).action,
            Action::RestartSteamVr
        );
        // Any other press cancels the confirmation.
        panel.click(button, &mut settings, &missing);
        assert!(panel.click([261.0, 101.0], &mut settings, &missing).changed);
        assert_eq!(
            panel.click(button, &mut settings, &missing).action,
            Action::None
        );
        let update = System::UpdateAvailable {
            version: "0.2.0".into(),
            failed: false,
        };
        assert_eq!(
            Panel::default()
                .click(button, &mut settings, &update)
                .action,
            Action::Update
        );
        let busy = System::Busy("UPDATING FRAME COMFORT...");
        assert_eq!(
            Panel::default().click(button, &mut settings, &busy),
            Click::default()
        );
        // Every footer line fits left of the button.
        for system in [
            missing,
            System::ObserverOutdated,
            update,
            System::UpdateAvailable {
                version: "10.20.30".into(),
                failed: true,
            },
            busy,
        ] {
            let image = Panel::default().render(&settings, "verified", &system);
            let row = |x: usize| image.pixels[615 * WIDTH + x];
            assert!((560..600).all(|x| row(x) == BG), "{system:?} overflows");
        }
    }

    #[test]
    fn every_game_source_status_has_its_own_panel_line() {
        use crate::source::Reason;
        let reasons = [
            Reason::NoSource,
            Reason::Ambiguous,
            Reason::UnsupportedLayout,
            Reason::InvalidDescription,
            Reason::IncompleteReport,
        ];
        let statuses = [
            "verified",
            "verified-next-movement",
            "learning",
            "no-fresh-source",
            "waiting-for-complete-frame",
            "separate-eye-pairing-unverified",
            "stereo-layout-unverified",
        ];
        for status in statuses.into_iter().chain(reasons.map(Reason::name)) {
            let line = source_status(status);
            assert_ne!(line, "GAME: NOT READY", "{status} has no specific text");
            assert!(line.len() <= 57, "{status} overflows the Mode panel");
        }
    }
}
