//! Release versions and the update state shared by the installer and the app.
//!
//! `frame-comfortctl` writes the update state to `update.json` in the
//! installation directory; the app reads it to show the panel's update line.
use crate::panel::System;
use serde_json::Value;
use std::{cmp::Ordering, fmt};

/// The version this build was made from.
pub const CURRENT: &str = env!("CARGO_PKG_VERSION");
/// File name of the update state, next to the installed app.
pub const UPDATE_FILE: &str = "update.json";
/// An update that has not reported back for this long has failed.
const UPDATE_STALE_SECS: u64 = 600;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Version(pub u32, pub u32, pub u32);

impl Version {
    /// Parses `1.2.3` or `v1.2.3`.
    pub fn parse(text: &str) -> Option<Self> {
        let mut parts = text.strip_prefix('v').unwrap_or(text).split('.');
        let mut next = || parts.next()?.parse().ok();
        let version = Self(next()?, next()?, next()?);
        parts.next().is_none().then_some(version)
    }

    pub fn current() -> Self {
        Self::parse(CURRENT).expect("the workspace version is MAJOR.MINOR.PATCH")
    }
}

impl fmt::Display for Version {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}.{}", self.0, self.1, self.2)
    }
}

/// What the update state means for a running app of version `running`, at
/// `now` (Unix seconds). States: `updating` (with `at`), `failed`,
/// `available`, `current`, `disabled`, `error`, each with `latest` where known.
pub fn update_notice(state: &Value, running: Version, now: u64) -> System {
    let latest = state["latest"]
        .as_str()
        .and_then(Version::parse)
        .filter(|latest| latest.cmp(&running) == Ordering::Greater);
    match (state["state"].as_str(), latest) {
        (Some("updating"), _)
            if state["at"]
                .as_u64()
                .is_some_and(|at| now.saturating_sub(at) < UPDATE_STALE_SECS) =>
        {
            System::Busy("UPDATING FRAME COMFORT...")
        }
        (Some("updating" | "failed"), Some(latest)) => System::UpdateAvailable {
            version: latest.to_string(),
            failed: true,
        },
        (Some("available"), Some(latest)) => System::UpdateAvailable {
            version: latest.to_string(),
            failed: false,
        },
        _ => System::Ready,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn versions_parse_and_order_numerically() {
        assert_eq!(Version::parse("v0.10.2"), Some(Version(0, 10, 2)));
        assert_eq!(Version::parse("1.2"), None);
        assert_eq!(Version::parse("1.2.3.4"), None);
        assert_eq!(Version::parse("1.2.x"), None);
        assert!(Version(0, 10, 0) > Version(0, 9, 9));
        assert_eq!(Version::current().to_string(), CURRENT);
    }

    #[test]
    fn update_states_map_to_panel_lines() {
        let running = Version(0, 1, 0);
        let notice = |state: Value| update_notice(&state, running, 1_000);
        assert_eq!(notice(json!({})), System::Ready);
        assert_eq!(
            notice(json!({"state": "available", "latest": "0.2.0"})),
            System::UpdateAvailable {
                version: "0.2.0".into(),
                failed: false
            }
        );
        // Already running the latest, or a downgrade: nothing to offer.
        assert_eq!(
            notice(json!({"state": "available", "latest": "0.1.0"})),
            System::Ready
        );
        assert_eq!(
            notice(json!({"state": "updating", "at": 900, "latest": "0.2.0"})),
            System::Busy("UPDATING FRAME COMFORT...")
        );
        // An update that never finished counts as failed.
        assert_eq!(
            notice(json!({"state": "updating", "at": 1, "latest": "0.2.0"})),
            System::UpdateAvailable {
                version: "0.2.0".into(),
                failed: true
            }
        );
        assert_eq!(
            notice(json!({"state": "failed", "latest": "0.2.0"})),
            System::UpdateAvailable {
                version: "0.2.0".into(),
                failed: true
            }
        );
        assert_eq!(
            notice(json!({"state": "error", "error": "offline"})),
            System::Ready
        );
    }
}
