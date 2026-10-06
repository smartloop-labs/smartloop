//! Port of `smartloop/trainer/__init__.py`.
//!
//! The Python package lazily re-exported `Trainer`, `TrainingProgress`,
//! `Builder`, `BuildProgress`, `TrainTask`, `BuildTask` and
//! `LoraGGUFConverter` from the optional `smartloop_trainer` distribution and
//! raised `ImportError` when it was absent.  The training stack has no Rust
//! port, so every entry point here reports it as not installed, with the same
//! message the Python import raised.

/// The message the Python lazy import raised when the trainer was missing.
pub const TRAINER_NOT_INSTALLED_MESSAGE: &str =
    "smartloop-trainer is not installed. Install with: make install-trainer";

/// The names the Python module exported (`__all__`).
pub const EXPORTS: [&str; 7] = [
    "Trainer",
    "TrainingProgress",
    "Builder",
    "BuildProgress",
    "TrainTask",
    "BuildTask",
    "LoraGGUFConverter",
];

/// Raised (returned) when an optional trainer component is requested.
#[derive(Debug, Clone, thiserror::Error, PartialEq, Eq)]
#[error("smartloop-trainer is not installed. Install with: make install-trainer")]
pub struct TrainerNotInstalled;

/// Fine-tunes a LoRA adapter for the current project (optional component).
#[derive(Debug)]
pub struct Trainer {
    _private: (),
}

impl Trainer {
    /// Always fails: the trainer is not part of this build.
    pub fn new() -> Result<Self, TrainerNotInstalled> {
        Err(TrainerNotInstalled)
    }
}

/// Builds/quantizes a GGUF from trained weights (optional component).
#[derive(Debug)]
pub struct Builder {
    _private: (),
}

impl Builder {
    /// Always fails: the builder is not part of this build.
    pub fn new() -> Result<Self, TrainerNotInstalled> {
        Err(TrainerNotInstalled)
    }
}

/// Look up an optional trainer export by name (Python `__getattr__`):
/// a known name yields [`TrainerNotInstalled`]; an unknown one yields an
/// `AttributeError`-style message.
pub fn import(name: &str) -> Result<(), String> {
    if EXPORTS.contains(&name) {
        Err(TRAINER_NOT_INSTALLED_MESSAGE.to_string())
    } else {
        Err(format!("module 'smartloop.trainer' has no attribute '{name}'"))
    }
}
