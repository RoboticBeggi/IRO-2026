#!/usr/bin/env bash
# Wi-Fi tuning for the Jetson's RTL8822CE card (Orin NX devkit). Run with sudo; safe to run again.
#
#     sudo tools/wifi_tune.sh            # country MY (Malaysia, where the robot is built)
#     sudo tools/wifi_tune.sh AT         # another country (use the one you are really in: legal limits)
#     tools/wifi_tune.sh status          # show the current state, no changes
#
# What it does (2 Nov 2026 findings):
#   1. Power save off. NetworkManager's default (powersave = 3) lets the radio doze between packets:
#      pings to the router jumped 2 -> 183 ms, SSH / VS Code feel laggy. Also the driver's own LPS
#      (rtw_power_mgnt, was 2) is turned off.
#   2. Country set. It was "00" (world), the strictest table: 5 GHz passive scan only, 20 dBm caps.
#   Both module options apply after a reboot (reloading the driver would cut the connection now).
# The robot competes offline: at the venue Wi-Fi can be switched off entirely: nmcli radio wifi off
set -euo pipefail
IF=$(iw dev | awk '/Interface/{print $2; exit}')

status() {
    echo "interface: $IF"
    iw dev "$IF" get power_save
    iw dev "$IF" link | grep -E "SSID|freq|signal|bitrate" || echo "  not connected"
    iw dev "$IF" info | grep -E "channel|txpower" || true
    echo "country: $(iw reg get | awk '/^country/{print $2; exit}')   driver power mgmt: $(cat /sys/module/rtl8822ce/parameters/rtw_power_mgnt 2>/dev/null || echo ?)"
}

if [ "${1:-}" = status ]; then status; exit 0; fi
[ "$(id -u)" -eq 0 ] || { echo "Run with sudo: sudo $0 ${*:-}"; exit 1; }
CC=${1:-MY}

# 1. power save off: now, and for every connection after reboots ("zz-" sorts after the stock file)
printf '[connection]\nwifi.powersave = 2\n' > /etc/NetworkManager/conf.d/zz-iro-wifi-powersave-off.conf
iw dev "$IF" set power_save off

# 2. country + driver power management (both read when the modules load, i.e. at boot)
printf 'options cfg80211 ieee80211_regdom=%s\noptions rtl8822ce rtw_country_code=%s rtw_power_mgnt=0\n' "$CC" "$CC" \
    > /etc/modprobe.d/iro-wifi.conf
iw reg set "$CC" || true

systemctl reload NetworkManager || true
echo "Done. Reboot once so the country and driver settings load. Now:"
status
