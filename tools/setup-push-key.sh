#!/usr/bin/env bash
# Give this Jetson permission to push backups to GitHub, using a deploy key
# that only works for this one repository.
# Run once per Jetson (or after reflashing). Then add the printed key at:
#   https://github.com/roboticbeggi/iro-2026/settings/keys  ->  Add deploy key
#   tick "Allow write access"
set -euo pipefail
KEY="$HOME/.ssh/iro2026_deploy"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

mkdir -p ~/.ssh && chmod 700 ~/.ssh
[ -f "$KEY" ] || ssh-keygen -t ed25519 -N "" -C "iro2026-$(hostname)" -f "$KEY" -q
if ! grep -q "Host github-iro2026" ~/.ssh/config 2>/dev/null; then
    cat >> ~/.ssh/config <<EOF

# IRO 2026 backups (deploy key for roboticbeggi/iro-2026)
Host github-iro2026
    HostName github.com
    User git
    IdentityFile $KEY
    IdentitiesOnly yes
EOF
    chmod 600 ~/.ssh/config
fi
grep -q "^github.com ssh-ed25519" ~/.ssh/known_hosts 2>/dev/null || \
    ssh-keyscan -t ed25519 github.com 2>/dev/null >> ~/.ssh/known_hosts

git -C "$ROOT" remote set-url --push origin git@github-iro2026:roboticbeggi/iro-2026.git

echo
echo "Add this key on GitHub (repo Settings > Deploy keys > Add deploy key, tick 'Allow write access'):"
echo
cat "$KEY.pub"
echo
echo "Then test with:  $ROOT/tools/backup.sh"
