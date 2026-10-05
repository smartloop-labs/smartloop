//! Port of `tests/conftest.py` (shared fixtures).

// ===== BEGIN unit: supervisor =====================================================
//
// Stand-in processes for the supervisor tests.  The Python tests ran small
// `python -c` / script children (a stub project agent, a stray agent, a probe);
// here the test binary re-executes itself into `supervisor_stub_entry` below,
// which plays the requested part.  The part is passed as a libtest filter
// argument (`slp-stub:<mode>:<arg>`) that matches no test, so the child runs
// exactly the one ignored entry point and nothing else.

/// Prefix of the argument that selects a stub's behaviour.
pub const SUPERVISOR_STUB_PREFIX: &str = "slp-stub:";

/// The libtest path of the stub entry point inside every target that mounts
/// this module as `shared`.
pub const SUPERVISOR_STUB_TEST: &str = "shared::supervisor_stub_entry";

/// argv that re-executes this test binary as a stub process.
///
/// Modes:
/// * `agent:<project_dir>` — the stub project agent: binds a loopback port,
///   answers every GET with 200 `{"status": "healthy", "path": ...}` and
///   echoes POST bodies, then publishes `agent.pid` / `agent.port` beside the
///   project and serves until killed;
/// * `wedged:<project_dir>` — publishes a port but serves nothing on it;
/// * `silent:<project_dir>` — publishes port `1` (nothing listens there);
/// * `probe:<path>` — writes its session/process-group/stdin/fd facts as JSON;
/// * `sleep:` — blocks until killed;
/// * `exit:<code>` — exits at once with `code`.
pub fn supervisor_stub_command(mode: &str, arg: &str) -> Vec<String> {
    let exe = std::env::current_exe().expect("test binary path");
    vec![
        exe.to_string_lossy().into_owned(),
        "--exact".into(),
        SUPERVISOR_STUB_TEST.into(),
        "--ignored".into(),
        "--nocapture".into(),
        "--test-threads=1".into(),
        "-q".into(),
        format!("{SUPERVISOR_STUB_PREFIX}{mode}:{arg}"),
    ]
}

/// `STUB_AGENT`: the stand-in project agent for `project_dir`.
pub fn supervisor_stub_agent(project_dir: &str) -> Vec<String> {
    supervisor_stub_command("agent", project_dir)
}

/// Spawn a stub as an ordinary child (the Python fixtures' `subprocess.Popen`).
pub fn supervisor_spawn_stub(mode: &str, arg: &str) -> std::process::Child {
    let command = supervisor_stub_command(mode, arg);
    std::process::Command::new(&command[0])
        .args(&command[1..])
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .expect("spawn stub")
}

/// A real process we can kill, standing in for a supervisor.
pub fn supervisor_sacrificial() -> std::process::Child {
    std::process::Command::new("sleep")
        .arg("300")
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .expect("spawn sleep")
}

/// Kill and reap `child`, ignoring errors (fixture teardown).
pub fn supervisor_reap(child: &mut std::process::Child) {
    let _ = child.kill();
    let _ = child.wait();
}

/// Poll `predicate` until it holds or `timeout` seconds pass.
pub fn supervisor_wait_for(timeout: f64, mut predicate: impl FnMut() -> bool) -> bool {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs_f64(timeout);
    while std::time::Instant::now() < deadline {
        if predicate() {
            return true;
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    predicate()
}

/// `threading.Event` for callbacks the tests hand to the code under test.
#[derive(Debug, Default)]
pub struct SupervisorFlag {
    flag: std::sync::Mutex<bool>,
    cv: std::sync::Condvar,
}

impl SupervisorFlag {
    pub fn new() -> std::sync::Arc<Self> {
        std::sync::Arc::new(Self::default())
    }

    pub fn set(&self) {
        *self.flag.lock().unwrap() = true;
        self.cv.notify_all();
    }

    pub fn is_set(&self) -> bool {
        *self.flag.lock().unwrap()
    }

    /// Wait up to `timeout` seconds; returns whether the flag is set.
    pub fn wait(&self, timeout: f64) -> bool {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs_f64(timeout);
        let mut flag = self.flag.lock().unwrap();
        while !*flag {
            let now = std::time::Instant::now();
            if now >= deadline {
                break;
            }
            flag = self.cv.wait_timeout(flag, deadline - now).unwrap().0;
        }
        *flag
    }
}

/// HTTP GET from a test, off any async runtime: the status code, or `None`.
pub fn supervisor_http_status(url: &str, timeout: f64) -> Option<u16> {
    let url = url.to_string();
    std::thread::spawn(move || {
        reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs_f64(timeout))
            .no_proxy()
            .build()
            .ok()?
            .get(&url)
            .send()
            .ok()
            .map(|r| r.status().as_u16())
    })
    .join()
    .ok()
    .flatten()
}

/// HTTP GET from a test, off any async runtime: `(status, body)`.
pub fn supervisor_http_get(url: &str, timeout: f64) -> Option<(u16, String)> {
    let url = url.to_string();
    std::thread::spawn(move || {
        let response = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs_f64(timeout))
            .no_proxy()
            .build()
            .ok()?
            .get(&url)
            .send()
            .ok()?;
        let status = response.status().as_u16();
        Some((status, response.text().ok()?))
    })
    .join()
    .ok()
    .flatten()
}

fn supervisor_stub_publish(project_dir: &str, pid: u32, port: u16) {
    std::fs::create_dir_all(project_dir).expect("project dir");
    std::fs::write(std::path::Path::new(project_dir).join("agent.pid"), pid.to_string()).unwrap();
    std::fs::write(std::path::Path::new(project_dir).join("agent.port"), port.to_string()).unwrap();
}

fn supervisor_stub_block() -> ! {
    loop {
        std::thread::park();
    }
}

fn supervisor_stub_serve(listener: std::net::TcpListener) {
    use std::io::{BufRead, BufReader, Read, Write};

    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(stream) = stream else { continue };
            std::thread::spawn(move || {
                let mut reader = BufReader::new(stream);
                let mut request_line = String::new();
                if reader.read_line(&mut request_line).is_err() {
                    return;
                }
                let mut length = 0usize;
                loop {
                    let mut line = String::new();
                    if reader.read_line(&mut line).unwrap_or(0) == 0 {
                        break;
                    }
                    let line = line.trim_end();
                    if line.is_empty() {
                        break;
                    }
                    if let Some((name, value)) = line.split_once(':') {
                        if name.trim().eq_ignore_ascii_case("content-length") {
                            length = value.trim().parse().unwrap_or(0);
                        }
                    }
                }
                let mut parts = request_line.split_whitespace();
                let method = parts.next().unwrap_or("").to_string();
                let path = parts.next().unwrap_or("").to_string();
                let body = if method == "POST" {
                    let mut body = vec![0u8; length];
                    let _ = reader.read_exact(&mut body);
                    body
                } else {
                    format!("{{\"status\": \"healthy\", \"path\": \"{path}\"}}").into_bytes()
                };
                let mut stream = reader.into_inner();
                let head = format!(
                    "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
                    body.len()
                );
                let _ = stream.write_all(head.as_bytes());
                let _ = stream.write_all(&body);
                let _ = stream.flush();
            });
        }
    });
}

fn supervisor_stub_probe(path: &str) {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        use std::os::unix::io::FromRawFd;

        // SAFETY: plain libc queries about this process.
        let (sid, pgid, pid) = unsafe { (libc::getsid(0), libc::getpgid(0), libc::getpid()) };
        // SAFETY: fd 0 is borrowed only for fstat and never closed.
        let stdin = std::mem::ManuallyDrop::new(unsafe { std::fs::File::from_raw_fd(0) });
        let devnull = std::fs::OpenOptions::new().read(true).write(true).open("/dev/null").unwrap();
        let stdin_devnull = stdin.metadata().map(|m| m.rdev()).ok() == devnull.metadata().map(|m| m.rdev()).ok();
        let mut fds: Vec<i64> = std::fs::read_dir("/dev/fd")
            .unwrap()
            .filter_map(|e| e.ok()?.file_name().to_str()?.parse().ok())
            .collect();
        fds.sort();
        let payload = serde_json::json!({
            "sid": sid, "pgid": pgid, "pid": pid,
            "stdin_devnull": stdin_devnull, "fds": fds,
        });
        std::fs::write(path, payload.to_string()).unwrap();
        drop(devnull);
    }
    #[cfg(not(unix))]
    {
        let _ = path;
    }
}

/// The stub process entry point (see [`supervisor_stub_command`]).  A no-op
/// when run without a stub argument.
#[test]
#[ignore = "helper process for the supervisor tests; runs only when re-executed"]
fn supervisor_stub_entry() {
    let Some(spec) = std::env::args().find_map(|a| a.strip_prefix(SUPERVISOR_STUB_PREFIX).map(str::to_string))
    else {
        return;
    };
    let (mode, arg) = spec.split_once(':').unwrap_or((spec.as_str(), ""));
    match mode {
        "agent" => {
            let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            let port = listener.local_addr().unwrap().port();
            supervisor_stub_serve(listener);
            supervisor_stub_publish(arg, std::process::id(), port);
            supervisor_stub_block();
        }
        "wedged" => {
            let port = {
                let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
                listener.local_addr().unwrap().port()
            };
            supervisor_stub_publish(arg, std::process::id(), port);
            supervisor_stub_block();
        }
        "silent" => {
            supervisor_stub_publish(arg, std::process::id(), 1);
            supervisor_stub_block();
        }
        "probe" => {
            supervisor_stub_probe(arg);
            std::process::exit(0);
        }
        "sleep" => supervisor_stub_block(),
        "exit" => std::process::exit(arg.parse().unwrap_or(1)),
        other => panic!("unknown stub mode {other}"),
    }
}

// ===== END unit: supervisor =======================================================
