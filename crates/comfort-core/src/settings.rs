use serde_json::{Map, Value, json};

pub const SETTINGS_VERSION: u32 = 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Periphery {
    Plain,
    Grid,
    Game,
}

impl Periphery {
    /// The settings file and status name.
    pub fn name(self) -> &'static str {
        match self {
            Self::Plain => "plain",
            Self::Grid => "grid",
            Self::Game => "game",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stick {
    Left,
    Right,
    Both,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OpeningShape {
    Circle,
    Rectangle,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SaveScope {
    All,
    Dashboard,
}

macro_rules! settings {
    ($($field:ident => ($key:literal, $min:literal, $max:literal, $default:literal, $dashboard:literal);)*) => {
        #[derive(Debug, Clone, PartialEq)]
        pub struct Settings {
            pub enabled: bool,
            pub periphery: Periphery,
            pub stick: Stick,
            pub shape: OpeningShape,
            $(pub $field: f32,)*
            pub rest_frame_face_size: u32,
            pub rest_frame_max_face_size: u32,
            pub dynamic_fov: bool,
            pub grid_over_ring: bool,
            pub crosshair: bool,
            pub nose: bool,
        }

        impl Default for Settings {
            fn default() -> Self {
                Self {
                    enabled: true,
                    periphery: Periphery::Game,
                    stick: Stick::Both,
                    shape: OpeningShape::Circle,
                    $($field: $default,)*
                    rest_frame_face_size: 0,
                    rest_frame_max_face_size: 816,
                    dynamic_fov: true,
                    grid_over_ring: false,
                    crosshair: false,
                    nose: false,
                }
            }
        }

        impl Settings {
            /// Clamp after all settings layers and command line overrides are applied.
            pub fn clamp(&mut self) {
                $(self.$field = if self.$field.is_finite() {
                    self.$field.clamp($min, $max)
                } else { $default };)*
                self.opening_min_width_deg = self.opening_min_width_deg.min(self.opening_width_deg);
                self.grid_brightness = self.grid_brightness.max(self.brightness);
                self.rest_frame_cross_fade_ms = self.rest_frame_cross_fade_ms.min(self.rest_frame_interval_ms * 0.5);
                if self.rest_frame_face_size != 0 {
                    self.rest_frame_face_size = self.rest_frame_face_size.clamp(64, 2048);
                }
                self.rest_frame_max_face_size = self.rest_frame_max_face_size.clamp(256, 2048);
            }

            /// Legacy unversioned files are ignored. Unknown versions are rejected.
            pub fn apply_json(&mut self, value: &Value) -> Result<(), &'static str> {
                let map = value.as_object().ok_or("settings must be a JSON object")?;
                if !map.contains_key("settingsVersion") {
                    return Ok(());
                }
                if !is_version_two(value) {
                    return Err("unsupported settings version");
                }
                self.apply_choices(map);
                $(if let Some(value) = map.get($key).and_then(Value::as_f64).filter(|v| v.is_finite()) {
                    self.$field = value.clamp(-(f32::MAX as f64), f32::MAX as f64) as f32;
                })*
                if let Some(size) = map.get("restFrameFaceSize").and_then(Value::as_f64).filter(|v| v.is_finite()) {
                    self.rest_frame_face_size = size.clamp(0.0, 4096.0) as u32;
                }
                if let Some(size) = map.get("restFrameMaxFaceSize").and_then(Value::as_f64).filter(|v| v.is_finite() && *v > 0.0) {
                    self.rest_frame_max_face_size = size.min(4096.0) as u32;
                }
                Ok(())
            }

            pub fn to_json(&self, scope: SaveScope) -> Value {
                let mut settings = self.clone();
                settings.clamp();
                let mut map = settings.choices_json();
                $(if $dashboard || scope == SaveScope::All {
                    map.insert($key.into(), json!(settings.$field));
                })*
                if scope == SaveScope::All {
                    map.insert("restFrameFaceSize".into(), json!(settings.rest_frame_face_size));
                    map.insert("restFrameMaxFaceSize".into(), json!(settings.rest_frame_max_face_size));
                }
                Value::Object(map)
            }
        }
    }
}

settings! {
    opening_width_deg => ("openingWidthDeg", 20.0, 140.0, 50.0, true);
    opening_height_deg => ("openingHeightDeg", 20.0, 120.0, 50.0, true);
    opening_min_width_deg => ("openingMinWidthDeg", 20.0, 140.0, 30.0, true);
    opening_fade => ("openingFade", 0.0, 0.6, 0.25, true);
    periphery_opacity => ("peripheryOpacity", 0.0, 1.0, 1.0, true);
    periphery_dim => ("peripheryDim", 0.0, 0.9, 0.0, true);
    deadzone => ("deadzone", 0.0, 0.95, 0.15, false);
    fade_in_ms => ("fadeInMs", 0.0, 3000.0, 120.0, true);
    fade_out_delay_ms => ("fadeOutDelayMs", 0.0, 3000.0, 500.0, true);
    fade_out_ms => ("fadeOutMs", 0.0, 3000.0, 300.0, true);
    refresh_blink_ms => ("refreshBlinkMs", 0.0, 400.0, 120.0, true);
    rest_frame_interval_ms => ("restFrameIntervalMs", 500.0, 3000.0, 1000.0, true);
    brightness => ("brightness", 0.03, 0.4, 0.125, true);
    grid_brightness => ("gridBrightness", 0.03, 0.6, 0.2, true);
    grid_thickness_cm => ("gridThicknessCm", 0.5, 10.0, 2.0, true);
    grid_spacing => ("gridSpacing", 0.25, 4.0, 2.0, true);
    crosshair_distance_m => ("crosshairDistanceM", 1.0, 10.0, 3.0, false);
    crosshair_size_deg => ("crosshairSizeDeg", 0.25, 5.0, 1.0, false);
    crosshair_opacity => ("crosshairOpacity", 0.0, 1.0, 0.5, false);
    nose_size => ("noseSize", 0.5, 2.0, 1.0, false);
    nose_opacity => ("noseOpacity", 0.0, 1.0, 0.6, false);
    rest_frame_cross_fade_ms => ("restFrameCrossFadeMs", 0.0, 500.0, 200.0, false);
    rest_frame_stillness_deg_per_sec => ("restFrameStillnessDegPerSec", 5.0, 90.0, 20.0, false);
    rest_frame_max_wait_ms => ("restFrameMaxWaitMs", 0.0, 500.0, 200.0, false);
    rest_frame_early_refresh_deg => ("restFrameEarlyRefreshDeg", 10.0, 90.0, 30.0, false);
    rest_frame_shell_half_size => ("restFrameShellHalfSize", 2.0, 5.5, 4.0, false);
    rest_frame_pose_offset_ms => ("restFramePoseOffsetMs", -50.0, 250.0, 0.0, false);
    // Measured on the Frame with a PC stream on November 14 (5 to 7 ms).
    rest_frame_stream_pose_offset_ms => ("restFrameStreamPoseOffsetMs", -50.0, 250.0, 6.0, false);
    rest_frame_move_limit_m => ("restFrameMoveLimitM", 0.1, 3.0, 1.0, false);
    rest_frame_mask_prediction_ms => ("restFrameMaskPredictionMs", -20.0, 50.0, 0.0, false);
}

pub fn is_version_two(value: &Value) -> bool {
    value.get("settingsVersion").and_then(Value::as_f64) == Some(f64::from(SETTINGS_VERSION))
}

impl Settings {
    fn apply_choices(&mut self, map: &Map<String, Value>) {
        self.periphery = match map.get("periphery").and_then(Value::as_str) {
            Some("plain") => Periphery::Plain,
            Some("grid") => Periphery::Grid,
            Some("game") => Periphery::Game,
            _ => self.periphery,
        };
        for key in ["hands", "restFrameTrigger", "stick"] {
            self.stick = match map.get(key).and_then(Value::as_str) {
                Some("left") => Stick::Left,
                Some("right") => Stick::Right,
                Some("both") => Stick::Both,
                _ => self.stick,
            };
        }
        self.shape = match map.get("shape").and_then(Value::as_str) {
            Some("circle") => OpeningShape::Circle,
            Some("rectangle") => OpeningShape::Rectangle,
            _ => self.shape,
        };
        for (key, target) in [
            ("dynamicFov", &mut self.dynamic_fov),
            ("gridOverRing", &mut self.grid_over_ring),
            ("crosshair", &mut self.crosshair),
            ("nose", &mut self.nose),
        ] {
            if let Some(value) = map.get(key).and_then(Value::as_bool) {
                *target = value;
            }
        }
    }

    fn choices_json(&self) -> Map<String, Value> {
        let periphery = self.periphery.name();
        let stick = match self.stick {
            Stick::Left => "left",
            Stick::Right => "right",
            Stick::Both => "both",
        };
        let shape = match self.shape {
            OpeningShape::Circle => "circle",
            OpeningShape::Rectangle => "rectangle",
        };
        [
            ("settingsVersion", json!(SETTINGS_VERSION)),
            ("periphery", json!(periphery)),
            ("stick", json!(stick)),
            ("shape", json!(shape)),
            ("dynamicFov", json!(self.dynamic_fov)),
            ("gridOverRing", json!(self.grid_over_ring)),
            ("crosshair", json!(self.crosshair)),
            ("nose", json!(self.nose)),
        ]
        .into_iter()
        .map(|(key, value)| (key.into(), value))
        .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn settings_layers_preserve_existing_values_and_clamp_after_overrides() {
        let mut settings = Settings::default();
        settings
            .apply_json(
                &json!({"settingsVersion": 2, "openingWidthDeg": 20, "openingMinWidthDeg": 45,
            "periphery": "plain", "crosshairSizeDeg": 2.5}),
            )
            .unwrap();
        settings.apply_json(&json!({"settingsVersion": 2, "openingWidthDeg": 60, "hands": "left", "stick": "both",
            "periphery": "unknown", "shape": 7, "dynamicFov": "false", "crosshair": true})).unwrap();
        settings.clamp();
        assert_eq!(settings.opening_width_deg, 60.0);
        assert_eq!(settings.opening_min_width_deg, 45.0);
        assert_eq!(settings.periphery, Periphery::Plain);
        assert_eq!(settings.stick, Stick::Both);
        assert_eq!(settings.crosshair_size_deg, 2.5);
        assert!(settings.dynamic_fov && settings.crosshair);

        let expected = settings.clone();
        assert!(
            settings
                .apply_json(&json!({"settingsVersion": 3, "openingWidthDeg": 80}))
                .is_err()
        );
        settings
            .apply_json(&json!({"periphery": "grid", "openingWidthDeg": 75}))
            .unwrap();
        assert_eq!(settings, expected);
    }

    #[test]
    fn settings_round_trip_keeps_file_only_values_out_of_dashboard_saves() {
        let mut settings = Settings::default();
        settings
            .apply_json(
                &json!({"settingsVersion": 2, "peripheryDim": 0.35, "restFrameFaceSize": 350,
            "restFrameIntervalMs": 500, "restFrameCrossFadeMs": 270, "openingHeightDeg": 1e100}),
            )
            .unwrap();
        settings.clamp();
        assert_eq!(settings.periphery_dim, 0.35);
        assert_eq!(settings.rest_frame_cross_fade_ms, 250.0);
        assert_eq!(settings.opening_height_deg, 120.0);
        let mut loaded = Settings::default();
        loaded
            .apply_json(&settings.to_json(SaveScope::All))
            .unwrap();
        assert_eq!(loaded, settings);

        let dashboard = settings.to_json(SaveScope::Dashboard);
        assert_eq!(dashboard["settingsVersion"], 2);
        assert!(dashboard.get("peripheryDim").is_some());
        for key in [
            "enabled",
            "deadzone",
            "restFrameFaceSize",
            "noseOpacity",
            "restFramePoseOffsetMs",
        ] {
            assert!(
                dashboard.get(key).is_none(),
                "{key} must not be saved by the dashboard"
            );
        }
    }
}
