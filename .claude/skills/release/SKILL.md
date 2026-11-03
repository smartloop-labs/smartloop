---
name: release
description: Release the slp agent/library from main — dispatch the Release workflow (patch/minor/major bump), merge the bot's "Release v<version>" PR, and wait for the tag build to publish binaries and the crate. Optionally continues into the Studio desktop release (/release desktop or "then release the desktop app"). Use when the user says to release, ship, or cut a version of the library/agent.
argument-hint: [patch|minor|major] [desktop]
disable-model-invocation: true
---

# release

Ship a new version of `slp` from `main`. Day-to-day PRs merge straight into `main`, so there is no `dev` → `main` step here (that is what the older `pr-release` skill does; `origin/dev` is stale).

Arguments: `$ARGUMENTS` — an optional bump type (`patch` default, `minor`, `major`), and `desktop` to continue into the Studio desktop release when the library is out.

All `gh` calls run as the **mehfuzh** account: prefix them with `GH_TOKEN=$(gh auth token --user mehfuzh)`. The active `gh` account is sometimes switched to `mehfuzheibuddy`, which cannot see `smartloop-ai` repos.

## How the pipeline works

`.github/workflows/release.yml` has two halves:

1. **Dispatch** (`workflow_dispatch`, input `bump`) — computes the next version from `Cargo.toml` `[package] version`, updates `Cargo.toml`, `Cargo.lock` (the `smartloop` entry), `install.sh` (`VERSION=`) and `install.ps1` (`$Version`), and opens PR **"Release v<version>"** from `release/v<version>` (opened by `github-actions[bot]`).
2. **Merge** — when a `release/v*` PR merges into `main`, it tags `v<version>` and dispatches `build.yml` with that tag, which builds every platform, uploads the archives to `https://dl.smartloop.ai/slp/<version>/`, and publishes the crate to the sparse registry at `dl.smartloop.ai/crates/` (never crates.io).

## Steps

1. **Preflight**
   - `gh pr list -R smartloop-ai/smartloop --base main --state open` — if a `release/v*` PR is already open, report it and stop (don't open a duplicate).
   - Read the current version from `main` (`gh api repos/smartloop-ai/smartloop/contents/Cargo.toml -q .content | base64 -d | grep -m1 '^version'`) and state the version this release will produce.
   - Make sure what's being shipped is on `main`: `git log v<current>..origin/main --oneline --no-merges`. If nothing, tell the user and stop.

2. **Dispatch the bump**
   - `gh workflow run release.yml -R smartloop-ai/smartloop -f bump=<bump>`
   - Wait for the `workflow_dispatch` run of `release.yml` to complete (`gh run list --workflow release.yml --event workflow_dispatch -L1`), then find the new PR.

3. **Review and merge the release PR**
   - `gh pr diff <n>` must change only the version in the four files above, old → new. Anything else: stop and show the user.
   - It has no CI checks (PRs opened by `GITHUB_TOKEN` don't trigger workflows) and its `pull_request` Build & Publish run may show "workflow file issue" with no jobs — both are expected; the code already passed CI in its own PRs.
   - `gh pr merge <n> --merge`.

4. **Watch the build**
   - The merge triggers `release.yml` (`pull_request` closed) → tag `v<version>` → `build.yml` (`workflow_dispatch`). Find that run: `gh run list --workflow build.yml --event workflow_dispatch -L1`.
   - Watch it in the background: `gh run watch <id> --exit-status --interval 30`. It takes a while (macOS/Linux/Windows builds + publish).
   - On failure: `gh run view <id> --log-failed | tail -50`, report the failing job, and stop. Do not continue to the desktop release.

5. **Verify it is downloadable** — every archive the desktop app fetches must return 200:
   ```
   for f in darwin-arm64-slp.tar.gz linux-amd64-slp.tar.gz linux-arm64-slp.tar.gz windows-amd64-slp.zip; do
     echo "$f $(curl -s -o /dev/null -w '%{http_code}' https://dl.smartloop.ai/slp/<version>/$f)"
   done
   ```

6. **Report** the version, tag, build run URL and the download check.

7. **Desktop (only if `desktop` was passed or the user asked)** — follow the `release-prod` skill in the Studio desktop repo (`~/source/studio-desktop/.claude/skills/release-prod/SKILL.md`) with `agent_version=<version>`. Only after step 5 passed: a desktop build pins the agent version and downloads it from `dl.smartloop.ai` at runtime.

Do not add AI attribution to anything the release creates.
