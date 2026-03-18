//! File watcher + background build for hot-reload.
//!
//! [`BuildWatcher`] manages background `cargo build` processes for the game
//! crate. It is a polling-based API — the editor calls [`BuildWatcher::poll`]
//! each frame to check if a triggered build has completed.

use std::collections::hash_map::DefaultHasher;
use std::collections::VecDeque;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};

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
/// 1. Create with the game crate directory and the project's scripts directory.
/// 2. Call [`trigger_build`] when source files change.
/// 3. Call [`poll`] each frame to check completion.
/// 4. Read [`state`] to get the result.
pub struct BuildWatcher {
    /// Path to the generated game crate directory (contains Cargo.toml).
    /// This is typically `.rkeditorcache/game/` inside the project.
    game_crate_dir: PathBuf,
    /// Path to the project's scripts directory (`assets/scripts/`).
    /// Source files here are the user-authored content checked for staleness.
    scripts_dir: PathBuf,
    /// Current build state.
    current_state: BuildState,
    /// Running child process, if any.
    child: Option<Child>,
    /// Last build error output, retained across state changes.
    last_error_output: Option<String>,
    /// Streaming stderr lines from the build process, pushed by a reader thread.
    stderr_lines: Arc<Mutex<VecDeque<String>>>,
    /// Handle to the stderr reader thread (joined on drop or process exit).
    _stderr_thread: Option<std::thread::JoinHandle<()>>,
}

impl BuildWatcher {
    /// Create a new build watcher.
    ///
    /// `game_crate_dir` is the generated crate (`.rkeditorcache/game/`).
    /// `scripts_dir` is the user's source directory (`assets/scripts/`).
    pub fn new(game_crate_dir: PathBuf, scripts_dir: PathBuf) -> Self {
        Self {
            game_crate_dir,
            scripts_dir,
            current_state: BuildState::Idle,
            child: None,
            last_error_output: None,
            stderr_lines: Arc::new(Mutex::new(VecDeque::new())),
            _stderr_thread: None,
        }
    }

    /// Current build state.
    pub fn state(&self) -> BuildState {
        self.current_state.clone()
    }

    /// Take the current build result, resetting to Idle if it was Success.
    ///
    /// Returns `Some(path)` if the build succeeded, `None` otherwise.
    /// This prevents the caller from re-processing the same result on the next frame.
    pub fn take_success(&mut self) -> Option<std::path::PathBuf> {
        if let BuildState::Success(ref path) = self.current_state {
            let path = path.clone();
            self.current_state = BuildState::Idle;
            Some(path)
        } else {
            None
        }
    }

    /// Reset to Idle. Use after consuming an Error state to stop re-reporting.
    pub fn reset(&mut self) {
        self.current_state = BuildState::Idle;
    }

    /// Drain any stderr lines accumulated since the last call.
    ///
    /// Returns an empty vec if no new lines are available. Call this each
    /// frame to get incremental build output for the console.
    pub fn drain_stderr_lines(&self) -> Vec<String> {
        if let Ok(mut buf) = self.stderr_lines.lock() {
            buf.drain(..).collect()
        } else {
            Vec::new()
        }
    }

    /// Spawn a background `cargo build` for the game crate.
    ///
    /// If a build is already in progress, this is a no-op (the caller should
    /// wait for the current build to finish before triggering another).
    ///
    /// If `console` is provided, stderr lines are pushed directly to it
    /// as they arrive (from a reader thread), classified by content.
    pub fn trigger_build(&mut self, console: Option<super::console::ConsoleBuffer>) {
        if matches!(self.current_state, BuildState::Compiling) {
            return;
        }

        let result = Command::new("cargo")
            .args(["build", "--lib", "--release"])
            .current_dir(&self.game_crate_dir)
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn();

        match result {
            Ok(mut child) => {
                // Spawn a thread to read stderr lines and push to console/buffer.
                let lines = Arc::clone(&self.stderr_lines);
                if let Ok(mut buf) = lines.lock() {
                    buf.clear();
                }
                let stderr = child.stderr.take();
                let handle = std::thread::spawn(move || {
                    use std::io::BufRead;
                    if let Some(stderr) = stderr {
                        let reader = std::io::BufReader::new(stderr);
                        for line in reader.lines() {
                            match line {
                                Ok(line) => {
                                    // Push to console immediately if available.
                                    if let Some(ref console) = console {
                                        let trimmed = line.trim();
                                        if !trimmed.is_empty() {
                                            if trimmed.starts_with("error") {
                                                console.error(trimmed);
                                            } else if trimmed.starts_with("warning") {
                                                console.warn(trimmed);
                                            } else if trimmed.starts_with("Compiling")
                                                || trimmed.starts_with("Finished")
                                                || trimmed.starts_with("Downloading")
                                                || trimmed.starts_with("Downloaded")
                                                || trimmed.starts_with("Blocking")
                                            {
                                                console.info(trimmed);
                                            }
                                        }
                                    }
                                    // Also buffer for error collection on failure.
                                    if let Ok(mut buf) = lines.lock() {
                                        buf.push_back(line);
                                    }
                                }
                                Err(_) => break,
                            }
                        }
                    }
                });
                self._stderr_thread = Some(handle);
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
                // Join the stderr reader thread so all lines are flushed.
                if let Some(handle) = self._stderr_thread.take() {
                    let _ = handle.join();
                }
                if status.success() {
                    // Try to extract dylib path from cargo JSON output.
                    let dylib_path = self.extract_dylib_path(&mut child);
                    self.current_state = BuildState::Success(dylib_path);
                } else {
                    // Collect all stderr lines from the streaming buffer.
                    let stderr = self.stderr_lines.lock()
                        .map(|buf| buf.iter().cloned().collect::<Vec<_>>().join("\n"))
                        .unwrap_or_default();
                    let stderr = if stderr.is_empty() {
                        "Build failed (no stderr output)".to_string()
                    } else {
                        stderr
                    };
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
    ///
    /// The dylib is built into the game crate's own `target/release/` directory
    /// (cargo's default when no `CARGO_TARGET_DIR` is set).
    pub fn expected_dylib_path(&self) -> PathBuf {
        let crate_name = self.crate_name().replace('-', "_");
        let filename = if cfg!(target_os = "linux") {
            format!("lib{crate_name}.so")
        } else if cfg!(target_os = "macos") {
            format!("lib{crate_name}.dylib")
        } else {
            format!("{crate_name}.dll")
        };

        self.game_crate_dir.join("target/release").join(&filename)
    }

    /// Check whether the existing dylib is stale (source files are newer).
    ///
    /// Returns `true` if a rebuild is needed:
    /// - The dylib does not exist, OR
    /// - Any `.rs` file in `scripts_dir` is newer than the dylib, OR
    /// - The `Cargo.toml` in the game crate is newer than the dylib.
    ///
    /// Returns `false` if the dylib exists and is up-to-date.
    pub fn needs_rebuild(&self) -> bool {
        let dylib_path = self.expected_dylib_path();
        let dylib_mtime = match std::fs::metadata(&dylib_path).and_then(|m| m.modified()) {
            Ok(t) => t,
            Err(e) => {
                log::debug!("needs_rebuild: dylib not found at {}: {e}", dylib_path.display());
                return true;
            }
        };

        // Check all .rs files under the scripts/source directory.
        if self.scripts_dir.is_dir() {
            if let Some(path) = find_newer_source(&self.scripts_dir, dylib_mtime) {
                log::debug!("needs_rebuild: source file newer than dylib: {}", path.display());
                return true;
            }
        }

        // Check the game crate's Cargo.toml (e.g., dependency path changes).
        let cargo_toml = self.game_crate_dir.join("Cargo.toml");
        if is_newer_than(&cargo_toml, dylib_mtime) {
            log::debug!("needs_rebuild: Cargo.toml newer than dylib");
            return true;
        }

        false
    }
}

/// Check if a file's mtime is newer than the reference time.
fn is_newer_than(path: &Path, reference: std::time::SystemTime) -> bool {
    match std::fs::metadata(path).and_then(|m| m.modified()) {
        Ok(t) => t > reference,
        Err(_) => false,
    }
}

/// Recursively check if any `.rs` file under `dir` is newer than `reference`.
/// Returns the path of the first newer file found, if any.
fn find_newer_source(dir: &Path, reference: std::time::SystemTime) -> Option<PathBuf> {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return None,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            if let Some(p) = find_newer_source(&path, reference) {
                return Some(p);
            }
        } else if path.extension().and_then(|e| e.to_str()) == Some("rs") {
            if is_newer_than(&path, reference) {
                return Some(path);
            }
        }
    }
    None
}

// ─── Content-hash based staleness check ─────────────────────────────────────

/// Name of the stamp file stored in the game crate directory.
const BUILD_STAMP_FILE: &str = ".build_stamp";

/// Compute a content hash of all `.rs` files under `dir` (recursively),
/// plus an optional extra file (e.g., Cargo.toml).
///
/// The hash incorporates file paths (sorted) and contents, so additions,
/// deletions, renames, and edits all produce a different hash.
pub fn hash_source_tree(dir: &Path, extra_file: Option<&Path>) -> u64 {
    let mut paths = Vec::new();
    collect_rs_files(dir, &mut paths);
    paths.sort();

    let mut hasher = DefaultHasher::new();

    // Hash extra file (e.g., Cargo.toml).
    if let Some(extra) = extra_file {
        if let Ok(content) = std::fs::read(extra) {
            extra.hash(&mut hasher);
            content.hash(&mut hasher);
        }
    }

    // Hash each .rs file by relative path + content.
    for path in &paths {
        path.hash(&mut hasher);
        if let Ok(content) = std::fs::read(path) {
            content.hash(&mut hasher);
        }
    }

    hasher.finish()
}

/// Recursively collect `.rs` file paths under `dir`.
fn collect_rs_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_rs_files(&path, out);
        } else if path.extension().and_then(|e| e.to_str()) == Some("rs") {
            out.push(path);
        }
    }
}

/// Read the stored build stamp (content hash) from the game crate directory.
///
/// Returns `None` if the stamp file doesn't exist or can't be parsed.
pub fn read_build_stamp(game_crate_dir: &Path) -> Option<u64> {
    let stamp_path = game_crate_dir.join(BUILD_STAMP_FILE);
    let text = std::fs::read_to_string(&stamp_path).ok()?;
    text.trim().parse::<u64>().ok()
}

/// Write the build stamp (content hash) to the game crate directory.
pub fn write_build_stamp(game_crate_dir: &Path, hash: u64) {
    let stamp_path = game_crate_dir.join(BUILD_STAMP_FILE);
    let _ = std::fs::write(&stamp_path, hash.to_string());
}

// ─── Cargo JSON helpers ─────────────────────────────────────────────────────

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
#[allow(dead_code)]
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
        let watcher = BuildWatcher::new(
            PathBuf::from("/tmp/nonexistent-crate-dir"),
            PathBuf::from("/tmp/nonexistent-scripts-dir"),
        );
        assert_eq!(watcher.state(), BuildState::Idle);
        assert!(watcher.last_error().is_none());
    }

    #[test]
    fn trigger_build_on_nonexistent_crate() {
        let mut watcher = BuildWatcher::new(
            PathBuf::from("/tmp/nonexistent-crate-dir-xyz"),
            PathBuf::from("/tmp/nonexistent-scripts-dir"),
        );
        watcher.trigger_build(None);
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
        let mut watcher = BuildWatcher::new(
            PathBuf::from("/tmp/nonexistent-crate-dir"),
            PathBuf::from("/tmp/nonexistent-scripts-dir"),
        );
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

    // ── hash/stamp tests ────────────────────────────────────────────

    #[test]
    fn hash_source_tree_deterministic() {
        let dir = std::env::temp_dir().join("rkf_test_hash_det");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("a.rs"), "fn a() {}").unwrap();
        std::fs::write(dir.join("b.rs"), "fn b() {}").unwrap();

        let h1 = hash_source_tree(&dir, None);
        let h2 = hash_source_tree(&dir, None);
        assert_eq!(h1, h2, "hash should be deterministic across calls");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn hash_source_tree_changes_on_edit() {
        let dir = std::env::temp_dir().join("rkf_test_hash_edit");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("a.rs"), "fn a() {}").unwrap();
        let h1 = hash_source_tree(&dir, None);

        std::fs::write(dir.join("a.rs"), "fn a() { 1 }").unwrap();
        let h2 = hash_source_tree(&dir, None);
        assert_ne!(h1, h2, "hash should change when file content changes");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn stamp_roundtrip() {
        let dir = std::env::temp_dir().join("rkf_test_stamp_rt");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        assert_eq!(read_build_stamp(&dir), None);
        write_build_stamp(&dir, 42);
        assert_eq!(read_build_stamp(&dir), Some(42));

        write_build_stamp(&dir, 9999999999);
        assert_eq!(read_build_stamp(&dir), Some(9999999999));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn hash_with_extra_file() {
        let dir = std::env::temp_dir().join("rkf_test_hash_extra");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("a.rs"), "fn a() {}").unwrap();
        let extra = dir.join("Cargo.toml");
        std::fs::write(&extra, "[package]\nname = \"test\"").unwrap();

        let h1 = hash_source_tree(&dir, Some(&extra));
        let h2 = hash_source_tree(&dir, None);
        assert_ne!(h1, h2, "extra file should affect hash");

        let h3 = hash_source_tree(&dir, Some(&extra));
        assert_eq!(h1, h3, "hash should be deterministic with extra file");

        let _ = std::fs::remove_dir_all(&dir);
    }
}
