//! Dynamic library loader for behavior system hot-reload.
//!
//! Loads cdylib plugins via `libloading`, resolves the `rkf_register` symbol
//! to register components/systems into a [`GameplayRegistry`], and performs
//! ABI version checking to detect mismatched compiler versions.

use std::ffi::CStr;
use std::path::Path;

use libloading::{Library, Symbol};

use super::GameplayRegistry;

/// ABI version string used to detect mismatched compiler versions between the
/// engine and a loaded dylib. Set to the crate's package version at compile time.
pub const ABI_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Errors that can occur when loading or interacting with a behavior dylib.
#[derive(Debug, Clone)]
pub enum DylibError {
    /// The dynamic library file could not be loaded (missing, bad format, etc.).
    LoadFailed(String),
    /// The required `rkf_register` symbol was not found in the library.
    SymbolNotFound(String),
    /// The `rkf_register` function returned an error or panicked.
    RegisterFailed(String),
    /// The dylib's ABI version does not match the engine's ABI version.
    AbiMismatch {
        /// The engine's ABI version.
        engine: String,
        /// The dylib's reported ABI version.
        dylib: String,
    },
}

impl std::fmt::Display for DylibError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DylibError::LoadFailed(msg) => write!(f, "failed to load dylib: {msg}"),
            DylibError::SymbolNotFound(msg) => write!(f, "symbol not found: {msg}"),
            DylibError::RegisterFailed(msg) => write!(f, "register failed: {msg}"),
            DylibError::AbiMismatch { engine, dylib } => {
                write!(
                    f,
                    "ABI version mismatch: engine={engine}, dylib={dylib}"
                )
            }
        }
    }
}

impl std::error::Error for DylibError {}

/// Wraps a loaded cdylib plugin for the behavior system.
///
/// On load, optionally checks ABI version compatibility. The caller then
/// invokes [`call_register`](DylibLoader::call_register) to populate a
/// [`GameplayRegistry`] with the plugin's components and systems.
pub struct DylibLoader {
    /// The loaded library handle. Named with underscore prefix because the
    /// value is held for its Drop side-effect (dlclose).
    _library: Library,
}

impl std::fmt::Debug for DylibLoader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DylibLoader").finish_non_exhaustive()
    }
}

impl DylibLoader {
    /// Load a cdylib from the given path.
    ///
    /// If the library exports an `rkf_abi_version` symbol, its return value is
    /// compared against [`ABI_VERSION`]. A mismatch produces
    /// [`DylibError::AbiMismatch`]. If the symbol is absent, a warning is logged
    /// but loading proceeds (backwards compatibility).
    pub fn load(path: &Path) -> Result<Self, DylibError> {
        // SAFETY: Loading a dynamic library can execute arbitrary code in its
        // init functions. We trust that the user is loading a known cdylib built
        // from the behavior system's plugin template.
        let library = unsafe { Library::new(path) }.map_err(|e| {
            DylibError::LoadFailed(format!("{}: {e}", path.display()))
        })?;

        // --- ABI version check (optional symbol) ---
        // SAFETY: We are resolving a simple extern "C" function that returns a
        // pointer to a static string. The symbol type is well-defined by the
        // plugin ABI contract.
        let abi_check: Result<Symbol<unsafe extern "C" fn() -> *const u8>, _> =
            unsafe { library.get(b"rkf_abi_version\0") };

        match abi_check {
            Ok(abi_fn) => {
                // SAFETY: The function is extern "C" and returns a pointer to a
                // null-terminated static string compiled into the dylib.
                let ptr = unsafe { abi_fn() };
                if ptr.is_null() {
                    log::warn!(
                        "dylib {:?}: rkf_abi_version returned null, skipping check",
                        path
                    );
                } else {
                    // SAFETY: The pointer is to a static null-terminated string
                    // in the dylib's data segment.
                    let version_cstr = unsafe { CStr::from_ptr(ptr as *const i8) };
                    let version = version_cstr.to_string_lossy();
                    if version != ABI_VERSION {
                        return Err(DylibError::AbiMismatch {
                            engine: ABI_VERSION.to_string(),
                            dylib: version.into_owned(),
                        });
                    }
                }
            }
            Err(_) => {
                log::warn!(
                    "dylib {:?}: no rkf_abi_version symbol, skipping ABI check",
                    path
                );
            }
        }

        Ok(DylibLoader { _library: library })
    }

    /// Resolve the `rkf_register` symbol and call it with the given registry.
    ///
    /// The symbol must be `extern "C" fn(&mut GameplayRegistry)`.
    pub fn call_register(&self, registry: &mut GameplayRegistry) -> Result<(), DylibError> {
        // SAFETY: We are resolving a well-known extern "C" symbol that the
        // plugin is required to export. The function signature is part of the
        // plugin ABI contract.
        let register_fn: Symbol<unsafe extern "C" fn(&mut GameplayRegistry)> =
            unsafe { self._library.get(b"rkf_register\0") }.map_err(|e| {
                DylibError::SymbolNotFound(format!("rkf_register: {e}"))
            })?;

        // SAFETY: The function pointer has the correct signature per the plugin
        // ABI. We pass a valid mutable reference to the registry.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            unsafe { register_fn(registry) };
        }));

        match result {
            Ok(()) => Ok(()),
            Err(panic) => {
                let msg = if let Some(s) = panic.downcast_ref::<&str>() {
                    s.to_string()
                } else if let Some(s) = panic.downcast_ref::<String>() {
                    s.clone()
                } else {
                    "unknown panic in rkf_register".to_string()
                };
                Err(DylibError::RegisterFailed(msg))
            }
        }
    }

    /// Explicitly unload the dynamic library.
    ///
    /// This is equivalent to dropping the `DylibLoader` — the underlying
    /// `libloading::Library` calls `dlclose` on drop.
    pub fn unload(self) {
        drop(self);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn dylib_error_display() {
        let load = DylibError::LoadFailed("not found".into());
        assert!(load.to_string().contains("not found"));
        assert!(load.to_string().contains("failed to load"));

        let sym = DylibError::SymbolNotFound("rkf_register".into());
        assert!(sym.to_string().contains("rkf_register"));
        assert!(sym.to_string().contains("symbol not found"));

        let reg = DylibError::RegisterFailed("panicked".into());
        assert!(reg.to_string().contains("panicked"));
        assert!(reg.to_string().contains("register failed"));

        let abi = DylibError::AbiMismatch {
            engine: "0.1.0".into(),
            dylib: "0.2.0".into(),
        };
        let abi_str = abi.to_string();
        assert!(abi_str.contains("0.1.0"));
        assert!(abi_str.contains("0.2.0"));
        assert!(abi_str.contains("mismatch"));
    }

    #[test]
    fn abi_version_is_set() {
        assert!(!ABI_VERSION.is_empty(), "ABI_VERSION must be non-empty");
        // Should match the crate version
        assert_eq!(ABI_VERSION, env!("CARGO_PKG_VERSION"));
    }

    #[test]
    fn load_nonexistent_fails() {
        let path = PathBuf::from("/tmp/nonexistent_rkf_plugin_12345.so");
        let result = DylibLoader::load(&path);
        assert!(result.is_err());
        match result.unwrap_err() {
            DylibError::LoadFailed(msg) => {
                assert!(msg.contains("nonexistent"), "error should mention path: {msg}");
            }
            other => panic!("expected LoadFailed, got: {other:?}"),
        }
    }

    #[test]
    fn unload_is_safe() {
        // Loading a nonexistent path fails, so we just verify the error path
        // doesn't panic on drop.
        let path = PathBuf::from("/tmp/nonexistent_rkf_plugin_67890.so");
        let result = DylibLoader::load(&path);
        assert!(result.is_err());
        // The Err variant drops cleanly — no Library to dlclose.
        drop(result);
    }
}
