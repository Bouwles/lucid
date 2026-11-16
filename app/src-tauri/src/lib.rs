//! LUCID desktop shell: starts the engine sidecar, hands its port and token to
//! the interface, launches games, and makes sure the engine never outlives the app.
//! The engine contract is in docs/engine-api.md.

use std::collections::VecDeque;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter, Manager, RunEvent, State, WindowEvent};

const READY_PREFIX: &str = "LUCID_ENGINE_READY ";
const READY_TIMEOUT: Duration = Duration::from_secs(60);
const STDERR_LINES: usize = 40;
/// How long the engine gets to exit after its stdin closes before it is killed.
const GRACEFUL_EXIT: Duration = Duration::from_secs(3);
/// Backstop so a smoke test can never hang CI, even if the page never reports.
const SMOKE_WATCHDOG: Duration = Duration::from_secs(150);

/// Set once an exit with a specific code is under way, so the window-closed
/// handler doesn't replace that code with 0.
static EXITING: AtomicBool = AtomicBool::new(false);

fn exit_with(app: &AppHandle, code: i32) {
    if !EXITING.swap(true, Ordering::SeqCst) {
        app.exit(code);
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EngineInfo {
    port: u16,
    token: String,
    version: String,
}

#[derive(Clone)]
enum Status {
    Starting,
    Ready(EngineInfo),
    Failed(String),
    Stopped,
}

struct Inner {
    status: Status,
    /// Bumped on every (re)start so threads from an old engine stand down.
    generation: u64,
    child: Option<Child>,
    /// Held open for the engine's lifetime. Closing it tells the engine to quit.
    stdin: Option<ChildStdin>,
}

pub struct Engine {
    inner: Mutex<Inner>,
    changed: Condvar,
    stderr_tail: Mutex<VecDeque<String>>,
    data_dir: PathBuf,
}

#[derive(Clone, Serialize)]
struct EngineExited {
    code: Option<i32>,
    stderr_tail: String,
}

#[derive(Clone, Serialize)]
struct GameExited {
    id: String,
    code: Option<i32>,
}

impl Engine {
    fn new(data_dir: PathBuf) -> Self {
        Self {
            inner: Mutex::new(Inner {
                status: Status::Stopped,
                generation: 0,
                child: None,
                stdin: None,
            }),
            changed: Condvar::new(),
            stderr_tail: Mutex::new(VecDeque::new()),
            data_dir,
        }
    }

    fn lock(&self) -> MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn tail(&self) -> String {
        let tail = self.stderr_tail.lock().unwrap_or_else(|e| e.into_inner());
        tail.iter().cloned().collect::<Vec<_>>().join("\n")
    }

    fn logs_path(&self) -> PathBuf {
        self.data_dir.join("logs").join("engine.log")
    }

    /// Change status only if this engine generation is still current and the
    /// status is one of the ones `from` accepts.
    fn transition(&self, generation: u64, from: fn(&Status) -> bool, to: Status) -> bool {
        let mut inner = self.lock();
        if inner.generation != generation || !from(&inner.status) {
            return false;
        }
        inner.status = to;
        self.changed.notify_all();
        true
    }

    fn wait_ready(&self, timeout: Duration) -> Result<EngineInfo, String> {
        let deadline = Instant::now() + timeout;
        let mut inner = self.lock();
        loop {
            match &inner.status {
                Status::Ready(info) => return Ok(info.clone()),
                Status::Failed(msg) => {
                    let tail = self.tail();
                    return Err(if tail.is_empty() { msg.clone() } else { format!("{msg}\n{tail}") });
                }
                Status::Stopped => return Err("The engine isn't running.".into()),
                Status::Starting => {
                    let now = Instant::now();
                    if now >= deadline {
                        return Err("Timed out waiting for the engine.".into());
                    }
                    inner = self
                        .changed
                        .wait_timeout(inner, deadline - now)
                        .unwrap_or_else(|e| e.into_inner())
                        .0;
                }
            }
        }
    }

    /// Close the engine's stdin, give it a moment to exit cleanly, then kill it.
    fn stop_locked(inner: &mut Inner) {
        inner.status = Status::Stopped;
        inner.stdin = None;
        let Some(mut child) = inner.child.take() else { return };
        let deadline = Instant::now() + GRACEFUL_EXIT;
        while Instant::now() < deadline {
            if let Ok(Some(_)) = child.try_wait() {
                return;
            }
            thread::sleep(Duration::from_millis(50));
        }
        // SIGTERM first: a PyInstaller one-file bootloader forwards it to the
        // real engine process, which SIGKILL would orphan.
        #[cfg(unix)]
        unsafe {
            libc::kill(child.id() as libc::pid_t, libc::SIGTERM);
        }
        let deadline = Instant::now() + Duration::from_secs(1);
        while Instant::now() < deadline {
            if let Ok(Some(_)) = child.try_wait() {
                return;
            }
            thread::sleep(Duration::from_millis(50));
        }
        let _ = child.kill();
        let _ = child.wait();
    }

    fn stop(&self) {
        let mut inner = self.lock();
        Self::stop_locked(&mut inner);
        self.changed.notify_all();
    }
}

/// The command that starts the engine. Debug builds run the Python source from
/// the repo; release builds run the bundled sidecar binary.
fn engine_command(app: &AppHandle, data_dir: &Path) -> Result<Command, String> {
    let (mut cmd, bundled_models) = if cfg!(debug_assertions) {
        let repo = std::env::var_os("LUCID_REPO")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../..")));
        let repo = repo
            .canonicalize()
            .map_err(|e| format!("Couldn't find the LUCID repo at {}: {e}", repo.display()))?;
        let models = repo.join("app").join("src-tauri").join("resources").join("models");
        // LUCID_ENGINE_BIN lets a debug build try the packaged engine binary.
        if let Some(bin) = std::env::var_os("LUCID_ENGINE_BIN") {
            (Command::new(bin), models)
        } else {
            let python = if cfg!(windows) {
                repo.join(".venv").join("Scripts").join("python.exe")
            } else {
                repo.join(".venv").join("bin").join("python")
            };
            let sep = if cfg!(windows) { ";" } else { ":" };
            let mut cmd = Command::new(python);
            cmd.args(["-m", "lucid_engine"])
                .env(
                    "PYTHONPATH",
                    format!("{}{sep}{}", repo.join("engine").display(), repo.display()),
                )
                .current_dir(&repo);
            (cmd, models)
        }
    } else {
        // Tauri bundles externalBin next to the app executable, without the target triple.
        let exe = std::env::current_exe().map_err(|e| e.to_string())?;
        let dir = exe.parent().ok_or("No executable directory")?;
        let bin = dir.join(format!("lucid-engine{}", std::env::consts::EXE_SUFFIX));
        let models = app
            .path()
            .resource_dir()
            .map_err(|e| e.to_string())?
            .join("resources")
            .join("models");
        (Command::new(bin), models)
    };
    cmd.arg("--data-dir")
        .arg(data_dir)
        .arg("--bundled-models")
        .arg(bundled_models);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
    Ok(cmd)
}

/// (Re)start the engine and watch it from background threads.
fn start_engine(app: &AppHandle, engine: &Arc<Engine>) {
    let mut inner = engine.lock();
    Engine::stop_locked(&mut inner);
    inner.generation += 1;
    let generation = inner.generation;
    inner.status = Status::Starting;
    engine.stderr_tail.lock().unwrap_or_else(|e| e.into_inner()).clear();
    engine.changed.notify_all();

    let spawned = engine_command(app, &engine.data_dir).and_then(|mut cmd| {
        cmd.stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("Couldn't start the engine: {e}"))
    });
    let mut child = match spawned {
        Ok(child) => child,
        Err(msg) => {
            inner.status = Status::Failed(msg);
            engine.changed.notify_all();
            return;
        }
    };
    let stdout = child.stdout.take();
    let stderr = child.stderr.take();
    inner.stdin = child.stdin.take();
    inner.child = Some(child);
    drop(inner);

    // stdout: wait for the ready line, then keep draining so the pipe never fills.
    if let Some(stdout) = stdout {
        let engine = engine.clone();
        thread::spawn(move || {
            for line in BufReader::new(stdout).lines() {
                let Ok(line) = line else { break };
                if let Some(json) = line.trim().strip_prefix(READY_PREFIX) {
                    let next = match serde_json::from_str::<EngineInfo>(json) {
                        Ok(info) => Status::Ready(info),
                        Err(e) => Status::Failed(format!("The engine's ready line didn't parse: {e}")),
                    };
                    engine.transition(generation, |s| matches!(s, Status::Starting), next);
                }
            }
        });
    }

    // stderr: keep the last few lines for error screens.
    if let Some(stderr) = stderr {
        let engine = engine.clone();
        thread::spawn(move || {
            for line in BufReader::new(stderr).lines() {
                let Ok(line) = line else { break };
                if cfg!(debug_assertions) {
                    eprintln!("[engine] {line}");
                }
                let mut tail = engine.stderr_tail.lock().unwrap_or_else(|e| e.into_inner());
                tail.push_back(line);
                while tail.len() > STDERR_LINES {
                    tail.pop_front();
                }
            }
        });
    }

    // Ready timeout.
    {
        let engine = engine.clone();
        thread::spawn(move || {
            thread::sleep(READY_TIMEOUT);
            let failed = engine.transition(
                generation,
                |s| matches!(s, Status::Starting),
                Status::Failed("The engine didn't say it was ready within 60 seconds.".into()),
            );
            if failed {
                let mut inner = engine.lock();
                if inner.generation == generation {
                    inner.stdin = None;
                    if let Some(child) = inner.child.as_mut() {
                        let _ = child.kill();
                    }
                }
            }
        });
    }

    // Exit watcher.
    {
        let engine = engine.clone();
        let app = app.clone();
        thread::spawn(move || loop {
            thread::sleep(Duration::from_millis(250));
            let mut inner = engine.lock();
            if inner.generation != generation {
                break;
            }
            let Some(child) = inner.child.as_mut() else { break };
            let code = match child.try_wait() {
                Ok(Some(status)) => status.code(),
                Ok(None) => continue,
                Err(_) => None,
            };
            inner.child = None;
            inner.stdin = None;
            let was_ready = matches!(inner.status, Status::Ready(_));
            if matches!(inner.status, Status::Starting | Status::Ready(_)) {
                // Give the stderr thread a moment to collect the last lines.
                drop(inner);
                thread::sleep(Duration::from_millis(150));
                let tail = engine.tail();
                let msg = match code {
                    Some(c) => format!("The engine exited with code {c}."),
                    None => "The engine was stopped by a signal.".to_string(),
                };
                engine.transition(
                    generation,
                    |s| matches!(s, Status::Starting | Status::Ready(_)),
                    Status::Failed(msg),
                );
                if was_ready {
                    let _ = app.emit("engine-exited", EngineExited { code, stderr_tail: tail });
                }
            }
            break;
        });
    }
}

// ------------------------------------------------------------------ commands

#[tauri::command]
async fn engine_info(engine: State<'_, Arc<Engine>>) -> Result<EngineInfo, String> {
    let engine = engine.inner().clone();
    tauri::async_runtime::spawn_blocking(move || engine.wait_ready(READY_TIMEOUT + Duration::from_secs(5)))
        .await
        .map_err(|e| e.to_string())?
}

#[tauri::command]
async fn retry_engine(app: AppHandle, engine: State<'_, Arc<Engine>>) -> Result<(), String> {
    let engine = engine.inner().clone();
    tauri::async_runtime::spawn_blocking(move || start_engine(&app, &engine))
        .await
        .map_err(|e| e.to_string())
}

#[tauri::command]
fn engine_logs_path(engine: State<'_, Arc<Engine>>) -> String {
    engine.logs_path().display().to_string()
}

#[tauri::command]
fn launch_game(app: AppHandle, id: String, path: String) -> Result<(), String> {
    let p = PathBuf::from(&path);
    if path.is_empty() || !p.exists() {
        return Err(format!("Couldn't find the game at '{path}'."));
    }
    // A macOS .app bundle is a folder; `open -W` waits until it quits.
    let mut cmd = if cfg!(target_os = "macos") && p.extension().is_some_and(|e| e == "app") {
        let mut c = Command::new("open");
        c.arg("-W").arg("-n").arg(&p);
        c
    } else {
        let mut c = Command::new(&p);
        if let Some(dir) = p.parent() {
            c.current_dir(dir);
        }
        c
    };
    let mut child = cmd
        .stdin(Stdio::null())
        .spawn()
        .map_err(|e| format!("Couldn't start the game: {e}"))?;
    thread::spawn(move || {
        let code = child.wait().ok().and_then(|s| s.code());
        let _ = app.emit("game-exited", GameExited { id, code });
    });
    Ok(())
}

#[tauri::command]
fn open_path(path: String) -> Result<(), String> {
    let p = PathBuf::from(&path);
    if !p.exists() {
        return Err(format!("'{path}' doesn't exist."));
    }
    let result = if cfg!(target_os = "macos") {
        Command::new("open").arg("-R").arg(&p).spawn()
    } else if cfg!(windows) {
        Command::new("explorer").arg(format!("/select,{}", p.display())).spawn()
    } else {
        let dir = if p.is_dir() { p.clone() } else { p.parent().map(Path::to_path_buf).unwrap_or(p.clone()) };
        Command::new("xdg-open").arg(dir).spawn()
    };
    result.map(|_| ()).map_err(|e| e.to_string())
}

#[tauri::command]
fn smoke_mode() -> bool {
    std::env::var("LUCID_SMOKE_TEST").is_ok_and(|v| v == "1")
}

fn write_smoke(engine: &Engine, ok: bool, detail: &str) {
    let path = engine.data_dir.join("logs").join("smoke.json");
    let body = serde_json::json!({
        "ok": ok,
        "detail": detail,
        "version": env!("CARGO_PKG_VERSION"),
        "unix_time": SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0),
    });
    let _ = std::fs::create_dir_all(path.parent().unwrap_or(Path::new(".")));
    let _ = std::fs::write(&path, serde_json::to_string_pretty(&body).unwrap_or_default());
    println!("LUCID_SMOKE {} {detail}", if ok { "ok" } else { "failed" });
}

#[tauri::command]
fn smoke_result(app: AppHandle, engine: State<'_, Arc<Engine>>, ok: bool, detail: String) {
    write_smoke(&engine, ok, &detail);
    engine.stop();
    exit_with(&app, if ok { 0 } else { 1 });
}

// ---------------------------------------------------------------------- app

pub fn run() {
    tauri::Builder::default()
        .setup(|app| {
            let data_dir = app
                .path()
                .data_dir()
                .map_err(|e| format!("No data folder: {e}"))?
                .join("LUCID");
            std::fs::create_dir_all(data_dir.join("logs"))?;
            let engine = Arc::new(Engine::new(data_dir));
            app.manage(engine.clone());

            let handle = app.handle().clone();
            thread::spawn(move || start_engine(&handle, &engine));

            if smoke_mode() {
                let handle = app.handle().clone();
                thread::spawn(move || {
                    thread::sleep(SMOKE_WATCHDOG);
                    let engine = handle.state::<Arc<Engine>>();
                    write_smoke(&engine, false, "the page never reported a result");
                    engine.stop();
                    exit_with(&handle, 1);
                });
            }
            Ok(())
        })
        .on_window_event(|window, event| {
            // Closing the main window quits LUCID on every platform, engine included.
            if let WindowEvent::Destroyed = event {
                if window.label() == "main" {
                    let app = window.app_handle();
                    app.state::<Arc<Engine>>().stop();
                    exit_with(app, 0);
                }
            }
        })
        .invoke_handler(tauri::generate_handler![
            engine_info,
            retry_engine,
            engine_logs_path,
            launch_game,
            open_path,
            smoke_mode,
            smoke_result
        ])
        .build(tauri::generate_context!())
        .expect("error while building LUCID")
        .run(|app, event| {
            if let RunEvent::Exit = event {
                app.state::<Arc<Engine>>().stop();
            }
        });
}
