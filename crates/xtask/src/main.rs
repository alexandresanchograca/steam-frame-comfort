//! Build utilities for the Cargo workspace.
#![forbid(unsafe_code)]

mod e2e;
mod package;

use std::{env, io, path::Path};

fn main() -> io::Result<()> {
    let args: Vec<_> = env::args().skip(1).collect();
    match args.as_slice() {
        [command, artifacts, output] if command == "package" => {
            package::package(Path::new(artifacts), Path::new(output))
        }
        [command, artifacts, output, flag, key] if command == "release" && flag == "--key" => {
            package::release(Path::new(artifacts), Path::new(output), Path::new(key))
        }
        [command, key] if command == "keygen" => package::keygen(Path::new(key)),
        [command, filters @ ..] if command == "e2e" => e2e::e2e(filters),
        _ => Err(io::Error::other(
            "usage: xtask package ARTIFACTS OUTPUT | xtask release ARTIFACTS OUTPUT --key KEY_FILE \
             | xtask keygen KEY_FILE | xtask e2e [TEST FILTER...]",
        )),
    }
}
