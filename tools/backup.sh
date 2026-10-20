#!/usr/bin/env bash
# Back up everything needed to rebuild the project off the Jetson.
# Works with or without internet, so it keeps protecting the work at the competition.
#
# 1. Copies state that lives outside the repo into backup/ (Claude memory
#    notes, LeRobot calibration from its default folder, package lists).
# 2. Snapshots the whole working tree, including uncommitted and untracked
#    files, as a commit on the branch autosave/<hostname>. Your current
#    branch, index and files are not touched.
# 3. Online: pushes the snapshot and all local branches to GitHub.
#    Offline: skips this in 3 seconds instead of hanging.
# 4. If the USB drive labelled IRO_BACKUP is plugged in (any mount point):
#    - a dated git bundle (full history, restorable with plain `git clone`),
#      keeping the newest 30
#    - a plain copy of the working tree, plus models/ and datasets/
#    - with --offline-kit: the Python environment and tools, so the robot can
#      be rebuilt with no internet at all
#
# Usage: tools/backup.sh [--offline-kit]      (or: iro backup)
# Runs every 10 min from cron (iro backup install). Log: ~/.iro-backup.log
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
BRANCH="autosave/$(hostname)"
LABEL="IRO_BACKUP"
STATE="$HOME/.iro-backup-state.json"
OFFLINE_KIT=0; [ "${1:-}" = "--offline-kit" ] && OFFLINE_KIT=1
ts() { date '+%Y-%m-%d %H:%M:%S'; }
# Record one field of the state file that `iro doctor` and `iro backup status` read
mark() { python3 - "$STATE" "$1" "$2" <<'EOF'
import json, sys
path, key, val = sys.argv[1:]
try: d = json.load(open(path))
except Exception: d = {}
d[key] = float(val) if val.replace('.', '', 1).isdigit() else val
json.dump(d, open(path, "w"), indent=1)
EOF
}

# Never run two backups at once (cron + manual)
exec 9>"$HOME/.iro-backup.lock"
flock -n 9 || { echo "$(ts) another backup is running, skipped"; exit 0; }

# --- 1. Outside state -> backup/
mkdir -p backup/system
MEM="$HOME/.claude/projects/$(echo "$ROOT" | sed 's#[/_.]#-#g')/memory"
[ -d "$MEM" ] && rsync -a --delete "$MEM/" backup/claude-memory/
DEFAULT_CAL="$HOME/.cache/huggingface/lerobot/calibration"
[ -d "$DEFAULT_CAL" ] && rsync -a "$DEFAULT_CAL/" backup/lerobot-calibration-default/
[ -x "$HOME/.venvs/iro/bin/python" ] && \
    "$HOME/.local/bin/uv" pip freeze --python "$HOME/.venvs/iro/bin/python" > env/requirements-lock.txt 2>/dev/null
apt-mark showmanual 2>/dev/null | sort > backup/system/apt-manual.txt
cat /etc/nv_tegra_release > backup/system/jetpack.txt 2>/dev/null
crontab -l > backup/system/crontab.txt 2>/dev/null

# Refuse files GitHub would reject; they belong on the USB drive.
big=(); skip=()
while IFS= read -r -d '' f; do
    if [ -f "$f" ] && [ "$(stat -c%s "$f")" -gt 52428800 ]; then
        big+=("$f"); skip+=(":(exclude,literal)$f")
    fi
done < <(git ls-files -z --others --modified --exclude-standard)
if [ ${#big[@]} -gt 0 ]; then
    echo "$(ts) WARNING: over 50 MB, not backed up to GitHub (add to .gitignore, keep on USB):" >&2
    printf '  %s\n' "${big[@]}" >&2
fi

# --- 2. Snapshot commit on autosave/<host>, using a private index
tmp_index="$(mktemp)"
cp .git/index "$tmp_index"
export GIT_INDEX_FILE="$tmp_index"
git add -A -- . "${skip[@]}"
tree=$(git write-tree)
unset GIT_INDEX_FILE
rm -f "$tmp_index"

# cron has no git identity unless one is configured; never fail because of it
git config user.email >/dev/null || export GIT_AUTHOR_EMAIL="autosave@$(hostname)" GIT_COMMITTER_EMAIL="autosave@$(hostname)"
git config user.name >/dev/null || export GIT_AUTHOR_NAME="IRO autosave" GIT_COMMITTER_NAME="IRO autosave"

head=$(git rev-parse HEAD)
parent=$(git rev-parse -q --verify "refs/heads/$BRANCH" || echo "$head")
parents=(-p "$parent")
# Also link the current branch tip, so the snapshot shows what it was based on
[ "$parent" != "$head" ] && ! git merge-base --is-ancestor "$head" "$parent" && parents+=(-p "$head")
if [ "$(git rev-parse "$parent^{tree}")" != "$tree" ]; then
    if ! commit=$(git commit-tree "$tree" "${parents[@]}" \
            -m "autosave $(ts) on $(git rev-parse --abbrev-ref HEAD)"); then
        echo "$(ts) ERROR: snapshot commit failed" >&2; exit 1
    fi
    git update-ref "refs/heads/$BRANCH" "$commit"
    echo "$(ts) snapshot $commit"
else
    echo "$(ts) no changes since last snapshot"
fi
mark last_snapshot "$(date +%s)"

# --- 3. Push, only if GitHub is reachable (fast fail when offline)
status=0
if timeout 3 bash -c 'exec 3<>/dev/tcp/github.com/22' 2>/dev/null; then
    export GIT_SSH_COMMAND="ssh -o ConnectTimeout=5 -o BatchMode=yes"
    for ref in $(git for-each-ref --format='%(refname:short)' refs/heads/); do
        if ! timeout 60 git push -q origin "$ref" 2>/dev/null; then
            echo "$(ts) push FAILED for $ref" >&2; status=1
        fi
    done
    if [ $status -eq 0 ]; then echo "$(ts) pushed to GitHub"; mark last_push_ok "$(date +%s)"; fi
else
    echo "$(ts) offline: GitHub push skipped"
fi

# --- 4. USB drive, found by its label wherever it is mounted
USB=$(findmnt -n -o TARGET -S "LABEL=$LABEL" 2>/dev/null | head -1)
if [ -n "$USB" ]; then
    dest="$USB/iro-2026-backup"
    mkdir -p "$dest/bundles"
    bundle="$dest/bundles/iro-2026_$(date +%Y%m%d-%H%M).bundle"
    if git bundle create -q "$bundle" --all 2>/dev/null; then
        # keep the newest 30 bundles
        ls -1t "$dest/bundles/"*.bundle | tail -n +31 | xargs -r rm -f
        rsync -a --delete --exclude '.git/' --exclude 'ros2_ws/build/' --exclude 'ros2_ws/install/' \
            --exclude 'ros2_ws/log/' "$ROOT/" "$dest/working-tree/"
        if [ $OFFLINE_KIT -eq 1 ]; then
            echo "$(ts) copying offline kit (Python env, tools); takes a few minutes the first time"
            rsync -a --delete "$HOME/.venvs/iro/" "$dest/offline-kit/venv-iro/"
            rsync -a --delete "$HOME/.local/opt/" "$dest/offline-kit/local-opt/"
            rsync -a --delete "$HOME/.arduino15/" "$dest/offline-kit/arduino15/"
            [ -d /var/cache/apt/archives ] && rsync -a --include '*.deb' --exclude '*' \
                /var/cache/apt/archives/ "$dest/offline-kit/apt-debs/" 2>/dev/null
            mark last_offline_kit "$(date +%s)"
        fi
        sync
        echo "$(ts) USB copy done: $bundle"
        mark last_usb_ok "$(date +%s)"
    else
        echo "$(ts) USB copy FAILED (drive full or read-only?)" >&2; status=1
    fi
fi
exit $status
