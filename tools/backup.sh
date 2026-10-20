#!/usr/bin/env bash
# Back up everything needed to rebuild the project off the Jetson.
#
# 1. Copies state that lives outside the repo into backup/ (Claude memory
#    notes, LeRobot calibration from its default folder, package lists).
# 2. Snapshots the whole working tree, including uncommitted and untracked
#    files, as a commit on the branch autosave/<hostname>. Your current
#    branch, index and files are not touched.
# 3. Pushes the snapshot and all local branches to GitHub.
# 4. If a USB drive is mounted at $IRO_USB (default /media/$USER/IRO_BACKUP),
#    mirrors the repo plus models/ and datasets/ there too.
#
# Run by hand any time (tools/backup.sh), and every 30 min from cron
# (tools/install-autobackup.sh). Log: ~/.iro-backup.log
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
BRANCH="autosave/$(hostname)"
USB="${IRO_USB:-/media/$USER/IRO_BACKUP}"
ts() { date '+%Y-%m-%d %H:%M:%S'; }

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

# --- 3. Push (each separately so one rejected branch does not block the rest)
status=0
for ref in $(git for-each-ref --format='%(refname:short)' refs/heads/); do
    if ! git push -q origin "$ref" 2>/dev/null; then
        echo "$(ts) push FAILED for $ref" >&2; status=1
    fi
done
[ $status -eq 0 ] && echo "$(ts) pushed to GitHub"

# --- 4. Optional second copy on a USB drive
if mountpoint -q "$USB" 2>/dev/null; then
    rsync -a --delete --exclude '.venv/' --exclude 'ros2_ws/build/' --exclude 'ros2_ws/install/' \
        "$ROOT/" "$USB/iro-2026/" && echo "$(ts) mirrored to $USB"
fi
exit $status
