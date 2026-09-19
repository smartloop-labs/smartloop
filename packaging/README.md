# Packaging: keeping `~/.smartloop` the home in every build

The framework home is `~/.smartloop` everywhere — App Store, Snap Store,
Developer ID, `pip install` — so the desktop app and the `slp` CLI operate on
one workspace instead of two. `smartloop/paths.py` resolves it: both store
sandboxes rewrite `$HOME` to a private directory, so the home is read from the
passwd record (which they do not rewrite) or from `SNAP_REAL_HOME`, never from
`$HOME`.

Resolving the path is not permission to open it. Each store needs its own
grant, below. Without the grant the app still starts — it keeps running out of
the sandboxed location, logs the denial, and the workspaces stay separate.

An install made by an older build already has its state inside the sandbox.
`adopt_confined_home` (`smartloop/workspace/migrations.py`) moves it out at
startup, before anything reads the home. It declines when `SLP_HOME` is set or
when a workspace already exists at `~/.smartloop`.

## Mac App Store

Sign with `entitlements.mas.plist`, not `entitlements.mac.plist` — the latter
is the Developer ID / hardened-runtime file and its `com.apple.security.cs.*`
keys are rejected on submission.

Access to `~/.smartloop` comes from
`com.apple.security.temporary-exception.files.home-relative-path.read-write`,
which **App Review requires a justification for and may refuse**. The
justification is that the app and a separately installed CLI are two
front-ends onto one local workspace of the user's own files.

If it is refused, the app can only keep its own workspace: spawn the backend
with `SLP_HOME` pointed at the container. Every home lookup checks `SLP_HOME`
first, so that one variable pins the whole framework.

```js
spawn(pythonBin, args, {
  env: { ...process.env, SLP_HOME: path.join(app.getPath('userData'), 'slp') },
});
```

## Snap Store

The auto-connecting `home` interface covers **non-hidden** files only, so it
does not reach `~/.smartloop`. That needs a `personal-files` plug:

```yaml
plugs:
  home:
    read: all
  dot-smartloop:
    interface: personal-files
    write:
      - $HOME/.smartloop
apps:
  smartloop:
    plugs: [home, network, network-bind, dot-smartloop]
```

`personal-files` does not auto-connect. Either request a snap declaration from
the store (a forum request under *store-requests*), or have users run:

```
snap connect smartloop:dot-smartloop
```

Until it is connected, reads and writes under `~/.smartloop` are denied by
AppArmor and the app runs out of `$SNAP_USER_DATA`.

## Electron

`os.homedir()` follows `$HOME` and so returns the sandboxed path in both
stores. `os.userInfo().homedir` goes to `getpwuid_r()` and does not. Resolve
once at startup, before any dependency is imported, and patch `os.homedir` so
libraries agree:

```js
const os = require('os');
const path = require('path');

const CONTAINER = '/Library/Containers/';

// Which sandbox, if any, $HOME has been rewritten to — checked before
// SNAP_REAL_HOME, because a snap hands its whole SNAP_* block to anything it
// launches (a terminal emulator starting a shell, say) and that child's $HOME
// means exactly what it says.
function confinement() {
  const home = os.homedir();
  if (home.includes(CONTAINER)) return 'mac';
  if (process.env.SNAP_NAME && /\/snap\/[^/]+\/[^/]+\/?$/.test(home)) return 'snap';
  return null;
}

function realHome() {
  if (!confinement()) return os.homedir();
  if (process.env.SNAP_REAL_HOME) return process.env.SNAP_REAL_HOME;
  try {
    const h = os.userInfo().homedir;   // getpwuid_r(), not $HOME
    if (h && path.isAbsolute(h) && !h.includes(CONTAINER)) return h;
  } catch {}
  return os.homedir();
}

const HOME = realHome();
os.homedir = () => HOME;
```

Do **not** overwrite `process.env.HOME`: Electron derives `userData`, caches
and crash dumps from it, and those belong in the container.
