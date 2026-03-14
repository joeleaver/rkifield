//! File watcher + background build for hot-reload.
//!
//! [`BuildWatcher`] manages background `cargo build` processes for the game
//! crate. It is a polling-based API — the editor calls [`BuildWatcher::poll`]
//! each frame to check if a triggered build has completed.

use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};

// ─── BuildState ─────────────────────────────────────────────────────────────

/// Current state of a background build.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BuildState {
    /// No build in progress, none completed.
    Idle,
    /// A `cargo build` is currently running.
    Compiling,
    /// Build succeeded — the built dylib path.
    Success(PathBuf),
    /// Build failed — stderr output.
    Error(String),
}

// ─── BuildWatcher ───────────────────────────────────────────────────────────

/// Manages background `cargo build` processes for the game crate.
///
/// Usage:
/// 1. Create with the game crate directory.
/// 2. Call [`trigger_build`] when source files change.
/// 3. Call [`poll`] each frame to check completion.
/// 4. Read [`state`] to get the result.
pub struct BuildWatcher {
    /// Path to the game crate directory (contains Cargo.toml).
    game_crate_dir: PathBuf,
    /// Current build state.
    current_state: BuildState,
    /// Running child process, if any.
    child: Option<Child>,
    /// Last build error output, retained across state changes.
    last_error_output: Option<String>,
}

impl BuildWatcher {
    /// Create a new build watcher for the given game crate directory.
    pub fn new(game_crate_dir: PathBuf) -> Self {
        Self {
            game_crate_dir,
            current_state: BuildState::Idle,
            child: None,
            last_error_output: None,
        }
    }

    /// Current build state.
    pub fn state(&self) -> BuildState {
        self.current_state.clone()
    }

    /// Spawn a background `cargo build` for the game crate.
    ///
    /// If a build is already in progress, this is a no-op (the caller should
    /// wait for the current build to finish before triggering another).
    pub fn trigger_build(&mut self) {
        if matches!(self.current_state, BuildState::Compiling) {
            return;
        }

        let crate_name = self.crate_name();

        // Use the game crate's own target directory to avoid contending with
        // the engine workspace's cargo lock (which may still be held by the
        // `cargo run` process that launched the editor).
        let target_dir = self.game_crate_dir.join("target");

        let result = Command::new("cargo")
            .args(["build", "-p", &crate_name, "--lib", "--message-format=json"])
            .current_dir(&self.game_crate_dir)
            .env("CARGO_TARGET_DIR", &target_dir)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn();

        match result {
            Ok(child) => {
                self.child = Some(child);
                self.current_state = BuildState::Compiling;
            }
            Err(e) => {
                let msg = format!("Failed to spawn cargo build: {e}");
                self.last_error_output = Some(msg.clone());
                self.current_state = BuildState::Error(msg);
            }
        }
    }

    /// Check if the background build has finished. Call this each frame.
    ///
    /// Updates [`state`] to `Success` or `Error` when the child process exits.
    pub fn poll(&mut self) {
        let child = match self.child.as_mut() {
            Some(c) => c,
            None => return,
        };

        match child.try_wait() {
            Ok(Some(status)) => {
                let mut child = self.child.take().unwrap();
                if status.success() {
                    // Try to extract dylib path from cargo JSON output.
                    let dylib_path = self.extract_dylib_path(&mut child);
                    self.current_state = BuildState::Success(dylib_path);
                } else {
                    let stderr = read_child_stderr(&mut child);
                    self.last_error_output = Some(stderr.clone());
                    self.current_state = BuildState::Error(stderr);
                }
            }
            Ok(None) => {
                // Still running.
            }
            Err(e) => {
                self.child = None;
                let msg = format!("Failed to poll cargo build: {e}");
                self.last_error_output = Some(msg.clone());
                self.current_state = BuildState::Error(msg);
            }
        }
    }

    /// Last build error output, if any.
    pub fn last_error(&self) -> Option<&str> {
        self.last_error_output.as_deref()
    }

    /// Derive the crate name from the directory name.
    fn crate_name(&self) -> String {
        self.game_crate_dir
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("game")
            .to_string()
    }

    /// Try to extract the built dylib path from cargo's JSON stdout.
    ///
    /// Falls back to constructing the expected path from the target directory.
    fn extract_dylib_path(&self, child: &mut Child) -> PathBuf {
        if let Some(stdout) = child.stdout.take() {
            let reader = std::io::BufReader::new(stdout);
            use std::io::BufRead;
            for line in reader.lines().flatten() {
                // Look for compiler artifact messages with dylib filenames.
                if line.contains("\"reason\":\"compiler-artifact\"")
                    && line.contains("\"crate_type\":[\"cdylib\"]")
                {
                    // Extract filenames from the JSON.
                    if let Some(path) = extract_dylib_from_json(&line) {
                        return path;
                    }
                }
            }
        }

        // Fallback: construct expected path.
        self.expected_dylib_path()
    }

    /// Construct the expected dylib path based on platform conventions.
    fn expected_dylib_path(&self) -> PathBuf {
        let crate_name = self.crate_name().replace('-', "_");
        let filename = if cfg!(target_os = "linux") {
            format!("lib{crate_name}.so")
        } else if cfg!(target_os = "macos") {
            format!("lib{crate_name}.dylib")
        } else {
            format!("{crate_name}.dll")
        };

        // Use the game crate's own target directory (set via CARGO_TARGET_DIR).
        self.game_crate_dir.join("target").join("debug").join(&filename)
    }
}

/// Extract a dylib path from a cargo JSON compiler-artifact message.
fn extract_dylib_from_json(json_line: &str) -> Option<PathBuf> {
    // Simple extraction: find "filenames":[...] and pick the .so/.dylib/.dll entry.
    let filenames_start = json_line.find("\"filenames\":[")?;
    let rest = &json_line[filenames_start..];
    let bracket_end = rest.find(']')?;
    let filenames_str = &rest[..bracket_end + 1];

    for part in filenames_str.split('"') {
        let p = part.trim();
        if p.ends_with(".so") || p.ends_with(".dylib") || p.ends_with(".dll") {
            return Some(PathBuf::from(p));
        }
    }
    None
}

/// Read stderr from a finished child process.
fn read_child_stderr(child: &mut Child) -> String {
    use std::io::Read;
    let mut stderr = String::new();
    if let Some(ref mut err) = child.stderr {
        let _ = err.read_to_string(&mut stderr);
    }
    if stderr.is_empty() {
        "Build failed (no stderr output)".to_string()
    } else {
        stderr
    }
}

// ─── CompileError ───────────────────────────────────────────────────────────

/// A single compiler error parsed from `rustc` / `cargo build` stderr output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompileError {
    /// The error message text.
    pub message: String,
    /// Source file path, if available.
    pub file: Option<String>,
    /// Line number in the source file, if available.
    pub line: Option<u32>,
    /// Column number in the source file, if available.
    pub column: Option<u32>,
}

/// Parse rustc-style error messages from cargo build stderr output.
///
/// Recognizes the format:
/// ```text
/// error[E0123]: some message
///  --> path/to/file.rs:42:10
/// ```
///
/// Also handles bare `error: message` without an error code.
/// Returns one [`CompileError`] per matched error block.
pub fn parse_cargo_errors(stderr: &str) -> Vec<CompileError> {
    let mut errors = Vec::new();
    let lines: Vec<&str> = stderr.lines().collect();
    let mut i = 0;

    while i < lines.len() {
        let line = lines[i];

        // Match "error[E0123]: message" or "error: message"
        let message = if let Some(rest) = line.strip_prefix("error[") {
            // error[E0123]: message
            if let Some(bracket_end) = rest.find("]: ") {
                Some(rest[bracket_end + 3..].to_string())
            } else {
                None
            }
        } else if let Some(rest) = line.strip_prefix("error: ") {
            Some(rest.to_string())
        } else {
            None
        };

        if let Some(msg) = message {
            // Look for " --> file:line:col" on the next line.
            let (file, line_num, col) = if i + 1 < lines.len() {
                parse_location(lines[i + 1])
            } else {
                (None, None, None)
            };

            errors.push(CompileError {
                message: msg,
                file,
                line: line_num,
                column: col,
            });
        }

        i += 1;
    }

    errors
}

/// Parse a " --> file:line:col" location line.
fn parse_location(line: &str) -> (Option<String>, Option<u32>, Option<u32>) {
    let trimmed = line.trim();
    let rest = match trimmed.strip_prefix("--> ") {
        Some(r) => r,
        None => return (None, None, None),
    };

    // Format: "path/to/file.rs:42:10"
    // Split from the right to handle paths with colons (e.g., Windows).
    let parts: Vec<&str> = rest.rsplitn(3, ':').collect();
    match parts.len() {
        3 => {
            let col = parts[0].parse::<u32>().ok();
            let line_num = parts[1].parse::<u32>().ok();
            let file = Some(parts[2].to_string());
            (file, line_num, col)
        }
        2 => {
            let line_num = parts[0].parse::<u32>().ok();
            let file = Some(parts[1].to_string());
            (file, line_num, None)
        }
        _ => (Some(rest.to_string()), None, None),
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_state_default_is_idle() {
        let watcher = BuildWatcher::new(PathBuf::from("/tmp/nonexistent-crate-dir"));
        assert_eq!(watcher.state(), BuildState::Idle);
        assert!(watcher.last_error().is_none());
    }

    #[test]
    fn trigger_build_on_nonexistent_crate() {
        let mut watcher = BuildWatcher::new(PathBuf::from("/tmp/nonexistent-crate-dir-xyz"));
        watcher.trigger_build();
        // The spawn itself may succeed (cargo starts, then fails) or fail
        // depending on the system. Either way, state should not be Idle.
        assert_ne!(watcher.state(), BuildState::Idle);

        // If it started compiling, poll until done.
        if matches!(watcher.state(), BuildState::Compiling) {
            // Wait for it to finish (it will fail quickly).
            loop {
                watcher.poll();
                if !matches!(watcher.state(), BuildState::Compiling) {
                    break;
                }
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
            // Should be an error since the crate doesn't exist.
            assert!(matches!(watcher.state(), BuildState::Error(_)));
        }
    }

    #[test]
    fn poll_when_idle_stays_idle() {
        let mut watcher = BuildWatcher::new(PathBuf::from("/tmp/nonexistent-crate-dir"));
        watcher.poll();
        assert_eq!(watcher.state(), BuildState::Idle);
        watcher.poll();
        assert_eq!(watcher.state(), BuildState::Idle);
    }

    // ── parse_cargo_errors tests ────────────────────────────────────────

    #[test]
    fn parse_cargo_errors_basic() {
        let stderr = "\
error[E0308]: mismatched types
 --> src/main.rs:42:10
  |
42 |     let x: i32 = \"hello\";
  |                  ^^^^^^^ expected `i32`, found `&str`
";
        let errors = parse_cargo_errors(stderr);
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].message, "mismatched types");
        assert_eq!(errors[0].file, Some("src/main.rs".to_string()));
        assert_eq!(errors[0].line, Some(42));
        assert_eq!(errors[0].column, Some(10));
    }

    #[test]
    fn parse_cargo_errors_multiple() {
        let stderr = "\
error[E0308]: mismatched types
 --> src/main.rs:10:5
  |
10 |     foo();
  |     ^^^^^ ...

error[E0425]: cannot find value `bar` in this scope
 --> src/lib.rs:20:9
  |
20 |     bar;
  |     ^^^ not found

error: aborting due to 2 previous errors
";
        let errors = parse_cargo_errors(stderr);
        // "error: aborting..." also matches — 3 total.
        assert_eq!(errors.len(), 3);

        assert_eq!(errors[0].message, "mismatched types");
        assert_eq!(errors[0].file, Some("src/main.rs".to_string()));
        assert_eq!(errors[0].line, Some(10));
        assert_eq!(errors[0].column, Some(5));

        assert_eq!(errors[1].message, "cannot find value `bar` in this scope");
        assert_eq!(errors[1].file, Some("src/lib.rs".to_string()));
        assert_eq!(errors[1].line, Some(20));
        assert_eq!(errors[1].column, Some(9));

        // The "aborting" line has no location.
        assert_eq!(errors[2].message, "aborting due to 2 previous errors");
        assert!(errors[2].file.is_none());
        assert!(errors[2].line.is_none());
    }

    #[test]
    fn parse_cargo_errors_no_errors() {
        let stderr = "warning: unused variable: `x`\n --> src/main.rs:5:9\n";
        let errors = parse_cargo_errors(stderr);
        assert!(errors.is_empty());
    }

    #[test]
    fn parse_cargo_errors_bare_error() {
        let stderr = "error: could not compile `my-crate`\n";
        let errors = parse_cargo_errors(stderr);
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].message, "could not compile `my-crate`");
        assert!(errors[0].file.is_none());
    }
}
