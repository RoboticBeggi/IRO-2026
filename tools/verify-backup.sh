#!/usr/bin/env bash
# Prove the backups restore: clone from the newest USB bundle and from GitHub
# into a temporary folder, and compare with the latest local snapshot.
# A backup you have never restored is only a hope.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BRANCH="autosave/$(hostname)"
want=$(git -C "$ROOT" rev-parse "$BRANCH^{tree}" 2>/dev/null) || { echo "No local snapshot yet: run iro backup"; exit 1; }
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
ok=0

check() {  # $1 = name, $2 = source to clone
    if ! git clone -q --no-checkout "$2" "$tmp/$1" 2>/dev/null; then
        echo "✖ $1: could not clone from $2"; return 1
    fi
    got=$(git -C "$tmp/$1" rev-parse "origin/$BRANCH^{tree}" 2>/dev/null || git -C "$tmp/$1" rev-parse "$BRANCH^{tree}" 2>/dev/null)
    git -C "$tmp/$1" checkout -q "${got:+$(git -C "$tmp/$1" rev-parse "origin/$BRANCH" 2>/dev/null || echo "$BRANCH")}" 2>/dev/null
    for f in arm/calibration/robots/so_follower config/robot.yaml tools/bootstrap.sh RECOVERY.md; do
        [ -e "$tmp/$1/$f" ] || { echo "✖ $1: restored copy is missing $f"; return 1; }
    done
    if [ "$got" = "$want" ]; then echo "✔ $1: restores, identical to the latest snapshot"
    else echo "⚠ $1: restores, but is older than the latest snapshot (run iro backup)"; fi
    ok=1
}

USB=$(findmnt -n -o TARGET -S LABEL=IRO_BACKUP 2>/dev/null | head -1)
if [ -n "$USB" ]; then
    newest=$(ls -1t "$USB"/iro-2026-backup/bundles/*.bundle 2>/dev/null | head -1)
    [ -n "$newest" ] && check usb "$newest" || echo "✖ usb: no bundle on $USB"
else
    echo "- usb: IRO_BACKUP drive not plugged in"
fi
if timeout 3 bash -c 'exec 3<>/dev/tcp/github.com/443' 2>/dev/null; then
    check github "$(git -C "$ROOT" remote get-url origin)"
else
    echo "- github: offline"
fi
[ $ok -eq 1 ] || exit 1
