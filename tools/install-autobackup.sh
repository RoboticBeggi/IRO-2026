#!/usr/bin/env bash
# Run tools/backup.sh every 30 minutes and 2 minutes after each boot, via your
# user crontab. Run once; running again does not add duplicates.
# Remove with:  crontab -l | grep -v iro-2026/tools/backup.sh | crontab -
set -euo pipefail
SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/backup.sh"
LOG="$HOME/.iro-backup.log"
{
    crontab -l 2>/dev/null | grep -v "$SCRIPT" || true
    echo "*/30 * * * * $SCRIPT >> $LOG 2>&1"
    echo "@reboot sleep 120 && $SCRIPT >> $LOG 2>&1"
} | crontab -
echo "Auto-backup installed. Check it with: tail $LOG"
