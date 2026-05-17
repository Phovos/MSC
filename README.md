---
"description": >
    'fosgit; a C/CPython baby no cry dual SCM system'
root: "./morphologi/fosgit"
name: "README.md"
version: 0.2.23
license-code+file(s): BSD 3-Clause
"© 2026 `Phovos` (phovos@outlook.com)":
    (1) https://gitlab.com/morphological/source/code
    (2) https://github.com/Morphological-Source-Code
    (3) https://reddit.com/r/morphological
---

# fosgit - fossmirror No tears dual-SCM

Automated, idempotent `fossil→git*n` mirroring with built-in audit trail (IN FOSSIL, which is chill [unlike so and so]) [do not let your infants eat this shampoo, ladies and gentlemen].

Let your fossil repository automatically push to GitLab, GitHub, Gitee, or any git remote on every commit, with fossil-native failure modes. Baby no cry, if dealing with git stresses you out just do everything perfectly and use this module, and no more tears. At least, when you mess up and have to spend a painful 30 minutes double and triple checking various CLI commands and args and their ever-loving flags; with full failure tracking via fossil tags. Fossil is chill about your failures, just logs them into the fossil repository, lazily. If for some reason you don't want fossil to artifice your failures you can, before your next fossil commit, get your git situation figured out, and then clean up the disaster zone such that fossil is none the wiser next time it is invoked.

## Why This Exists

I love Fossil SCM's simplicity and power, but increasing my power level in a world that has a society requires talking to git, and I ain't happy about it.
- GitLab/GitHub/Aliyun/Gitee etc. hosting
- CI/CD pipelines that only speak git
- Collaboration with git-only teams
- (Bonus!) Backup redundancy

The problem: Existing solutions are brittle, stateful, or require manual intervention.

The solution: Use fossil's own tag system as audit ledger. Every export and push is recorded as a fossil tag — no external database, no state files, just clean, self-documenting history.

## Features

✓ Idempotent — Safe to run multiple times, syncs only what's new  
✓ Automated — Triggers on every fossil commit via hooks  
✓ Multi-remote — Push to unlimited git remotes simultaneously  
✓ Failure-aware — Failed pushes are tagged, retried automatically  
✓ Zero config — Edit one file, run one command  
✓ Audit trail — `fossil tag list` shows complete sync history  
✓ Tested — test suite included

## Architecture (easy, for babies/grandmas)

```
fossil commit
    ↓
fossil hook fires
    ↓
fossilgitmirror (C binary)
    ├─→ 1. fossil git export (incremental)
    ├─→ 2. tag fossil: gitexport:<hash>
    └─→ 3. for each remote:
            ├─→ git push <remote>
            └─→ tag fossil: gitpush:<remote>:ok/fail:<hash>
```

Audit tags (visible via `fossil tag list`):
- `gitexport:a3f9c1...` — Export completed, git hash recorded
- `gitpush:gitlab:ok:a3f9c1...` — Push to gitlab succeeded
- `gitpush:github:ok:a3f9c1...` — Push to github succeeded
- `gitpush:gitee:fail:a3f9c1...` — Push to gitee failed (fix + recommit retries)

## Installation

### Prerequisites

- Fossil SCM (`fossil`)
- Git (`git`)
- C compiler (`gcc` or `clang`)
- Python 3.6+

### Setup

1. Create config directory:
   ```bash
   mkdir -p ~/.config/fossil-mirror
   cd ~/.config/fossil-mirror
   ```

2. Place files:
   ```bash
   # Copy these files to ~/.config/fossil-mirror/:
   - fossilgitmirror.c
   - fossilgitinit.py
   - remotes           # your config (see template below)
   - test_mirror.sh    # optional, for testing
   ```

3. Edit `remotes` config:
   ```bash
   # Example ~/.config/fossil-mirror/remotes

   mirror  ~/projects/myproject-git-mirror
   fossil  ~/projects/myproject.fossil

   # Compiler (uncomment for your platform):
   linux:   gcc -std=c99 -O2 -Wall -Wextra
   # mac:     clang -std=c99 -O2 -Wall -Wextra

   # Git remotes (add as many as you need):
   # GitLab is the native/primary remote for this project:
   gitlab   git@gitlab.com:youruser/myproject.git
   github   git@github.com:youruser/myproject.git
   gitee    git@gitee.com:youruser/myproject.git
   ```

4. Run setup:
   ```bash
   chmod +x fossilgitinit.py
   ./fossilgitinit.py
   ```

   This will:
   - Compile `fossilgitmirror.c`
   - Perform initial export
   - Configure git remotes
   - Install fossil hook
   - Create `.initialized` sentinel

5. Test (optional but recommended):
   ```bash
   chmod +x test_mirror.sh
   ./test_mirror.sh
   ```

## Usage

### Normal Operation

Just commit to fossil. The mirror happens automatically:

```bash
fossil commit -m "Add feature"
# → Hook fires
# → Export to git
# → Push to all remotes
# → Tags recorded
```

### Check Sync Status

```bash
fossil tag list | grep git
```

Example output:
```
gitexport:a3f9c1d8...
gitpush:gitlab:ok:a3f9c1d8...
gitpush:github:ok:a3f9c1d8...
gitpush:gitee:fail:a3f9c1d8...  ← needs attention
```

### Manual Sync

If you need to sync without committing:

```bash
~/.config/fossil-mirror/fossilgitmirror
```

### Handling Failures

When a push fails (non-fast-forward, auth, network), you'll see:

```
┌─ PUSH FAILED ───────────────────────────────────────────────┐
│ Remote: gitee                                                │
│                                                              │
│ This usually means:                                          │
│   • Non-fast-forward (remote has commits you don't)          │
│   • Authentication failed                                    │
│   • Network issue                                            │
│                                                              │
│ To fix:                                                      │
│   1. Inspect: cd ~/projects/myproject-git-mirror             │
│   2. Debug:   git push gitee --all -v                        │
│   3. Resolve manually (fetch/merge/rebase as needed)         │
│   4. Next fossil commit will retry automatically             │
└──────────────────────────────────────────────────────────────┘
```

Fix it:
```bash
cd ~/projects/myproject-git-mirror
git fetch gitee
git merge gitee/main  # or rebase, resolve conflicts
# Don't push manually! Just commit to fossil:
cd ~/projects/myproject
fossil commit -m "Merge upstream changes"
# → Hook retries all failed remotes
```

## Configuration

### Config File Format

`~/.config/fossil-mirror/remotes`:

```
# Comments start with #
# Blank lines are ignored

# Required:
mirror <path>      # Where to maintain the git mirror
fossil <path>      # Path to your .fossil file

# Platform-specific compiler (pick ONE):
linux: <compiler command>
mac: <compiler command>
windows: <compiler command>

# Git remotes (name + URL):
<name> <url>
<name> <url>
...
```

### Adding a Remote

Edit `~/.config/fossil-mirror/remotes`:

```bash
# Add new remote:
gitlab   git@gitlab.com:youruser/myproject.git
```

Then sync to activate:
```bash
~/.config/fossil-mirror/fossilgitmirror
```

The remote is wired automatically on next run.

### Removing a Remote

Remove the line from config. The remote stays in git but won't be pushed to anymore.

To fully remove it from git:
```bash
cd ~/projects/myproject-git-mirror
git remote remove gitee
```

## Troubleshooting

### "Hook not firing"

Check the hook is installed:
```bash
fossil hook list -R ~/projects/myproject.fossil
```

Should show:
```
after-commit: /home/user/.config/fossil-mirror/fossilgitmirror
```

> **Note:** Fossil uses `after-commit` hooks, not `after-receive` (that's a git hook name).

### "Compilation fails"

Check your compiler:
```bash
gcc --version    # Linux
clang --version  # macOS
```

Update the compiler line in `remotes` if needed.

### "Push fails repeatedly"

Common causes:

1. SSH key not configured — test each remote:
   ```bash
   ssh -T git@gitlab.com   # GitLab (native)
   ssh -T git@github.com   # GitHub
   ssh -T gitee.com        # Gitee
   ```

2. Remote diverged:
   ```bash
   cd ~/projects/myproject-git-mirror
   git fetch --all
   git log --oneline --graph --all
   ```

3. Wrong URL:
   ```bash
   git remote -v  # check URLs
   ```

### "Export takes forever"

Initial export of large repos can be slow. Subsequent exports are incremental (fast).

For huge repos (>10k commits), consider:
- Run initial export manually before installing hook
- Use `fossil git export --incremental`

### "Git mirror out of sync"

Nuclear option — rebuild from scratch:

```bash
rm -rf ~/projects/myproject-git-mirror
fossil git export ~/projects/myproject-git-mirror -R ~/projects/myproject.fossil
# Then configure remotes again
```

## Advanced

### Multiple Fossil Repos

Each fossil repo needs its own config:

```bash
~/.config/fossil-mirror-projectA/remotes
~/.config/fossil-mirror-projectB/remotes
```

Modify `fossilgitinit.py` or create wrapper scripts.

### CI/CD Integration

The git mirror is a normal git repo. GitLab is the native CI target — point your pipeline at the mirror:

```yaml
# .gitlab-ci.yml  (lives in the FOSSIL repo, CI reads from the git mirror)
stages:
  - test

build-and-test:
  stage: test
  image: ubuntu:latest
  script:
    - make test
```

Because the fossil hook pushes to the GitLab remote on every commit, GitLab CI triggers automatically off the mirror. No manual push needed.

For GitHub Actions on the GitHub mirror:

```yaml
# .github/workflows/ci.yml  (if you also want GitHub CI)
on: [push]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: make test
```

### Custom Hook Logic

Edit `fossilgitmirror.c` to add:
- Slack notifications
- Deployment triggers
- Branch-specific handling
- Tag filtering

Recompile:
```bash
cd ~/.config/fossil-mirror
gcc -std=c99 -O2 fossilgitmirror.c -o fossilgitmirror
```

## Uninstallation

1. Remove hook:
   ```bash
   fossil hook delete -R ~/projects/myproject.fossil after-commit
   ```

2. Delete files:
   ```bash
   rm -rf ~/.config/fossil-mirror
   ```

3. Keep or delete git mirror:
   ```bash
   # Keep it — it's just a git repo now, nothing special to do.

   # Delete it:
   rm -rf ~/projects/myproject-git-mirror
   ```

## Design Philosophy

### Why Fossil Tags?

Fossil's tag system is:
- Distributed — Every clone has the complete tag history
- Immutable — Tags are part of the repository artifact tree
- Queryable — `fossil tag list` is instant, no parsing needed
- Append-only — Failed pushes don't erase history

This makes it perfect for audit trails.

### Why C for the Sync Binary?

- Fast — Runs on every commit, needs to be instant
- Portable — C99 + POSIX runs everywhere fossil runs
- No dependencies — One binary, no pip/npm/gem hell
- Predictable — No GC pauses, no runtime versions

### Why Python for Init?

- Runs once — Speed doesn't matter
- Platform detection — `platform` module simplifies OS detection
- Error handling — Better error messages than shell

### Why Not a Fossil Extension?

Fossil extensions are great, but:
- Hooks need to run post-commit
- External binary is simpler to debug
- Easier to customize without patching fossil

## Credits

Inspired by the eternal struggle of loving Fossil but needing git (because, people).

Built with:
- Fossil SCM's `fossil git export`

## License

BSD 3-Clause. Built entirely from public-domain-friendly components — thank you to `[fossil-scm.org](https://fossil-scm.org)`, home of a key SQLite developer and one of the finest pieces of software engineering around.

## Contributing

This is a personal tool that solves a specific problem. If you find bugs or have improvements:

1. Test them thoroughly
2. Keep it simple
3. Submit clear explanations that respect the design decisions and impeccable taste required to develop no-deps, idempotent, idiomatic, small-to-purpose repositories
4. Or just fork and 'forgettaboutit'. ('I'm workin' here!')

---

The goal is not full automation of everything. It's rock-solid, 'baby no cry' automation of the clean path, and clear systematization of the messy parts. If people behave, maybe I'll make a 'baby no cry' button that doesn't fix git pushing issues for you — it makes git pushing issues no longer a problem by reverting and undoing the failed commit and staging, re-running fossilgitmirror against the new state (which does include the errors from the failed state currently in scope; delete those manually if you don't want error messages to fail-forward and self-artifice).
