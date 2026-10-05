//! Server lifecycle management for Smartloop.
//!
//! Port of `smartloop/server.py`.  Provides [`start_server`], [`stop_server`]
//! and [`get_status`] as importable functions.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use once_cell::sync::Lazy;
use serde_json::{Map, Value};

use crate::api::start_server_foreground;
use crate::config::AppSettings;
use crate::supervisor::registry::process_alive;

const LOG: &str = "smartloop.server";

/// `<home_dir>/server.pid`, resolved once from [`AppSettings`].
pub static PID_FILE: Lazy<PathBuf> =
    Lazy::new(|| PathBuf::from(AppSettings::new().home_dir()).join("server.pid"));

/// Get the base URL for the API server.
pub fn get_server_url(host: &str, port: u16) -> String {
    format!("http://{host}:{port}")
}

/// `GET url` with a `timeout_secs` budget, answering the status code and the
/// raw body.
///
/// A blocking reqwest client may not be driven from inside a tokio runtime
/// (it builds and drops one of its own), so when called from async context
/// the request runs on a scratch thread.
fn http_get(url: &str, timeout_secs: u64) -> Result<(u16, Vec<u8>), String> {
    let fetch = {
        let url = url.to_string();
        move || -> Result<(u16, Vec<u8>), String> {
            let client = reqwest::blocking::Client::builder()
                .timeout(Duration::from_secs(timeout_secs))
                .build()
                .map_err(|e| e.to_string())?;
            let response = client.get(&url).send().map_err(|e| e.to_string())?;
            let status = response.status().as_u16();
            let body = response.bytes().map_err(|e| e.to_string())?;
            Ok((status, body.to_vec()))
        }
    };
    if tokio::runtime::Handle::try_current().is_ok() {
        std::thread::spawn(fetch).join().unwrap_or_else(|_| Err("request thread panicked".to_string()))
    } else {
        fetch()
    }
}

/// `httpx.get(url).json()`: the decoded body, or `None` on a transport error
/// or a body that is not JSON.
fn http_get_json(url: &str, timeout_secs: u64) -> Option<Value> {
    let (_, body) = http_get(url, timeout_secs).ok()?;
    serde_json::from_slice(&body).ok()
}

/// Check if the API server is already running (`GET /health` == 200, 2s timeout).
pub fn is_server_running(host: &str, port: u16) -> bool {
    matches!(http_get(&format!("{}/health", get_server_url(host, port)), 2), Ok((200, _)))
}

/// Read the PID from the pid file if it exists and the process is alive;
/// a stale file is removed.
///
/// Liveness goes through [`process_alive`], not `kill(pid, 0)`: on Windows a
/// "signal" terminates the process, so the usual check would kill the running
/// agent every time someone asked for its status.
pub fn read_pid_file() -> Option<u32> {
    if !PID_FILE.exists() {
        return None;
    }
    // An unreadable file or one that does not hold an integer reads as no pid
    // (and is left in place).
    let text = std::fs::read_to_string(&*PID_FILE).ok()?;
    let pid: i64 = text.trim().parse().ok()?;
    if process_alive(Some(pid)) {
        return u32::try_from(pid).ok();
    }
    let _ = std::fs::remove_file(&*PID_FILE);
    None
}

/// Write `pid` to the pid file, creating its directory.
pub fn write_pid_file(pid: u32) -> std::io::Result<()> {
    if let Some(parent) = PID_FILE.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&*PID_FILE, pid.to_string())
}

/// Poll until `pid` is gone, for at most `timeout`.  True when it exited.
fn wait_for_exit(pid: u32, timeout: Duration) -> bool {
    let deadline = Instant::now() + timeout;
    loop {
        if !process_alive(Some(i64::from(pid))) {
            return true;
        }
        if Instant::now() >= deadline {
            return false;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
}

/// Deliver `signal` to `pid` through sysinfo (which falls back to a hard kill
/// where the signal does not exist, as on Windows).
fn signal_process(pid: u32, signal: sysinfo::Signal) -> Result<(), String> {
    use sysinfo::{Pid, ProcessRefreshKind, ProcessesToUpdate, System};

    let sys_pid = Pid::from_u32(pid);
    let mut system = System::new();
    system.refresh_processes_specifics(ProcessesToUpdate::Some(&[sys_pid]), true, ProcessRefreshKind::nothing());
    let process = system.process(sys_pid).ok_or_else(|| format!("process no longer exists (pid={pid})"))?;
    let delivered = match process.kill_with(signal) {
        Some(delivered) => delivered,
        None => process.kill(),
    };
    if delivered {
        Ok(())
    } else {
        Err(format!("could not signal process (pid={pid})"))
    }
}

/// Ask, then insist: SIGTERM, wait 10s, SIGKILL, wait 5s.
fn terminate_process(pid: u32) -> Result<(), String> {
    signal_process(pid, sysinfo::Signal::Term)?;
    if wait_for_exit(pid, Duration::from_secs(10)) {
        return Ok(());
    }
    signal_process(pid, sysinfo::Signal::Kill)?;
    if wait_for_exit(pid, Duration::from_secs(5)) {
        return Ok(());
    }
    Err(format!("timeout after 5 seconds (pid={pid})"))
}

/// Stop the background API server if running (terminate, wait 10s, then
/// kill).  Returns true if the server was stopped.
pub fn stop_server() -> bool {
    match read_pid_file() {
        Some(pid) if pid != 0 => match terminate_process(pid) {
            Ok(()) => {
                let _ = std::fs::remove_file(&*PID_FILE);
                tracing::info!(target: LOG, "Server stopped (PID {pid})");
                true
            }
            Err(e) => {
                tracing::error!(target: LOG, "Could not stop server: {e}");
                let _ = std::fs::remove_file(&*PID_FILE);
                false
            }
        },
        _ => {
            tracing::info!(target: LOG, "No server running");
            false
        }
    }
}

/// Python truthiness of a JSON value.
fn truthy(value: Option<&Value>) -> bool {
    match value {
        None | Some(Value::Null) => false,
        Some(Value::Bool(b)) => *b,
        Some(Value::Number(n)) => n.as_f64().is_some_and(|f| f != 0.0),
        Some(Value::String(s)) => !s.is_empty(),
        Some(Value::Array(a)) => !a.is_empty(),
        Some(Value::Object(o)) => !o.is_empty(),
    }
}

/// `data.get("projects", [])` as a list of objects.
fn projects_of(data: &Value) -> Vec<&Map<String, Value>> {
    data.get("projects")
        .and_then(Value::as_array)
        .map(|projects| projects.iter().filter_map(Value::as_object).collect())
        .unwrap_or_default()
}

/// `dict.get(key)` → the value or `null`.
fn get_or_null(map: &Map<String, Value>, key: &str) -> Value {
    map.get(key).cloned().unwrap_or(Value::Null)
}

/// Return server status as a JSON object.
///
/// Always has `"running"`; when running also `host`, `port`, `pid`,
/// `model_loaded`, `model_name`, optional model stats, `active_project`, and
/// under a supervisor `supervised`, `supervisor`, `model_host`, `agents`.
pub fn get_status(host: &str, port: u16) -> Value {
    let not_running = || serde_json::json!({"running": false});

    if port == 0 {
        return not_running();
    }

    if !is_server_running(host, port) {
        return not_running();
    }

    let pid = read_pid_file();
    let mut result = Map::new();
    result.insert("running".into(), Value::Bool(true));
    result.insert("host".into(), Value::from(host));
    result.insert("port".into(), Value::from(port));
    result.insert("pid".into(), pid.map(Value::from).unwrap_or(Value::Null));

    let base = get_server_url(host, port);

    if let Some(Value::Object(health)) = http_get_json(&format!("{base}/health"), 5) {
        result.insert("model_loaded".into(), health.get("model_loaded").cloned().unwrap_or(Value::Bool(false)));
        result.insert("model_name".into(), get_or_null(&health, "model_name"));
        for key in ["quantization", "n_ctx", "n_gpu_layers", "flash_attn", "model_size_bytes", "memory_percent"] {
            if let Some(value) = health.get(key).filter(|v| !v.is_null()) {
                result.insert(key.into(), value.clone());
            }
        }
    }

    if let Some(data) = http_get_json(&format!("{base}/v1/projects"), 10) {
        let current = projects_of(&data).into_iter().find(|p| truthy(p.get("current")));
        if let Some(current) = current {
            result.insert(
                "active_project".into(),
                serde_json::json!({
                    "id": get_or_null(current, "id"),
                    "name": get_or_null(current, "name"),
                    "model_name": get_or_null(current, "model_name"),
                }),
            );
        }
    }

    // Under a supervisor there is a process per project rather than one
    // server, so status is a tree: the supervisor, the model host, then a row
    // per project.  Absent under the --no-supervised fallback, where /health
    // already says everything there is to say.
    if let Some(Value::Object(agents)) = http_get_json(&format!("{base}/agents"), 5) {
        if truthy(agents.get("supervised")) {
            let names = project_names(host, port);
            let mut rows = match agents.get("agents") {
                Some(Value::Array(rows)) => rows.clone(),
                _ => Vec::new(),
            };
            for agent in rows.iter_mut() {
                if let Value::Object(agent) = agent {
                    let name = agent
                        .get("project_id")
                        .and_then(Value::as_str)
                        .and_then(|id| names.get(id).cloned())
                        .unwrap_or(Value::Null);
                    agent.insert("name".into(), name);
                }
            }
            result.insert("supervised".into(), Value::Bool(true));
            result.insert(
                "supervisor".into(),
                agents.get("supervisor").cloned().unwrap_or_else(|| Value::Object(Map::new())),
            );
            result.insert(
                "model_host".into(),
                agents.get("model_host").cloned().unwrap_or_else(|| Value::Object(Map::new())),
            );
            result.insert("agents".into(), Value::Array(rows));
        }
    }

    Value::Object(result)
}

/// `{project_id: name}`, so the status table reads in project names.
fn project_names(host: &str, port: u16) -> Map<String, Value> {
    let Some(data) = http_get_json(&format!("{}/v1/projects", get_server_url(host, port)), 10) else {
        return Map::new();
    };
    projects_of(&data)
        .into_iter()
        .filter_map(|p| {
            let id = p.get("id").and_then(Value::as_str)?;
            Some((id.to_string(), get_or_null(p, "name")))
        })
        .collect()
}

/// Start the API server in the foreground (blocking).
///
/// Sets `SLP_SUPERVISOR` to `"1"`/`"0"` from `supervised`, then runs the
/// server.  `debug` is accepted for signature compatibility and unused, as in
/// Python.
///
/// * `service` — auto-restart on crash with exponential backoff.
/// * `supervised` — run a process per project behind this port, with the
///   model factored out into a host they share (the default).  The client
///   contract is unchanged either way; what changes is the blast radius of
///   one project's runaway job.
pub fn start_server(
    host: &str,
    port: u16,
    debug: bool,
    service: bool,
    supervised: bool,
) -> anyhow::Result<()> {
    let _ = debug;
    // SAFETY: called from the CLI's main thread before the server (and any of
    // its threads) starts.
    unsafe {
        std::env::set_var("SLP_SUPERVISOR", if supervised { "1" } else { "0" });
    }

    start_server_foreground(host, port, service)
}
