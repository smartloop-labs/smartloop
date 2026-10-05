//! Where the user's home actually is, once a store's sandbox has moved it.
//!
//! Port of `smartloop/paths.py`.  `$HOME` is rewritten by the macOS App
//! Sandbox (`~/Library/Containers/<bundle-id>/Data`) and by snapd under strict
//! confinement (`~/snap/<snap-name>/<revision>`).  Neither rewrites the passwd
//! database, and snapd >= 2.46 states the real home in `SNAP_REAL_HOME`.
//! This module answers *where*; the caller still has to cope with a denial.

/// How macOS spells the App Sandbox container it rewrites `$HOME` to.
pub const SANDBOX_HOME_MARKER: &str = "/Library/Containers/";

/// How snapd spells `$SNAP_USER_DATA`: `<real home>/snap/<snap-name>/<revision>`.
pub const SNAP_HOME_RE: &str = r"^(?P<home>.+)/snap/[^/]+/[^/]+/?$";

/// The real home `home` is a sandbox directory inside, or `None` when `$HOME`
/// is not a confinement directory and is to be taken at its word.
fn unconfined_parent(home: &str) -> Option<String> {
    let _ = home;
    todo!()
}

/// The passwd record's home directory for the current uid, if readable.
pub fn passwd_home() -> Option<String> {
    todo!()
}

/// The user's home, reading past a store sandbox's rewrite of `$HOME`.
pub fn real_home() -> String {
    todo!()
}

/// The sandbox directory `$HOME` was rewritten to, or `None` off a store build.
pub fn confined_home() -> Option<String> {
    todo!()
}

/// `~`-expand `path` against [`real_home`].
///
/// `~otheruser/...` names somebody else's home and is resolved through the
/// passwd database, like `os.path.expanduser`.
pub fn expand_home(path: &str) -> String {
    expand_home_with(path, real_home)
}

/// [`expand_home`] with an injectable real-home resolver (test seam standing
/// in for monkeypatching `smartloop.paths.real_home`).
pub fn expand_home_with(path: &str, real_home: impl Fn() -> String) -> String {
    let _ = (path, real_home);
    todo!()
}

/// `os.path.expanduser` equivalent: `~` / `~/x` against `$HOME` (falling
/// back to passwd), `~user/x` against that user's passwd record; anything
/// unresolvable is returned unchanged.
pub fn expanduser(path: &str) -> String {
    let _ = path;
    todo!()
}
