use std::path::PathBuf;

fn main() {
    // Tauri refuses to build if an externalBin is missing. The real engine
    // binary is built separately (engine/ with PyInstaller), so a fresh clone
    // gets a stub that exits with an explanation. Debug builds don't use it.
    let target = std::env::var("TARGET").unwrap_or_default();
    let ext = if target.contains("windows") { ".exe" } else { "" };
    let bin = PathBuf::from("binaries").join(format!("lucid-engine-{target}{ext}"));
    if !bin.exists() {
        let _ = std::fs::create_dir_all("binaries");
        let stub = "#!/bin/sh\necho 'This is a placeholder. Build the LUCID engine binary first (see docs/app.md).' >&2\nexit 1\n";
        if std::fs::write(&bin, stub).is_ok() {
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                let _ = std::fs::set_permissions(&bin, std::fs::Permissions::from_mode(0o755));
            }
            println!(
                "cargo:warning=No engine binary at {}; wrote a placeholder. Release builds need the real one.",
                bin.display()
            );
        }
    }
    tauri_build::build()
}
