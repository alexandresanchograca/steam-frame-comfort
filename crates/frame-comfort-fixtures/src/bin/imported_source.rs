#[cfg(target_os = "linux")]
fn main() -> Result<std::process::ExitCode, Box<dyn std::error::Error>> {
    use std::{
        io::{self, BufRead, Write},
        process::ExitCode,
    };

    let dma = match std::env::args().nth(1).as_deref() {
        Some("--opaque-fd") => false,
        Some("--dma-buf") => true,
        _ => return Err("usage: frame-comfort-source-producer --opaque-fd|--dma-buf".into()),
    };
    let available = comfort_runtime::vulkan::fixture::publish_imported_source(dma, || {
        // The consumer may acquire the image only after this marker.
        println!("source ready");
        io::stdout().flush()?;
        io::stdin().lock().read_line(&mut String::new())?;
        Ok(())
    })?;
    if available {
        Ok(ExitCode::SUCCESS)
    } else {
        println!("unsupported: the selected Vulkan device cannot import and export this image");
        Ok(ExitCode::from(77))
    }
}

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("The source producer requires Linux.");
    std::process::exit(1);
}
