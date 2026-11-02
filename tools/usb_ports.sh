#!/usr/bin/env bash
# Live map of the Jetson's 4 USB-A sockets (behind the on-board Realtek hub: Linux ports 1-2.1 ... 1-2.4,
# their USB 3 halves 2-1.1 ... 2-1.4). Unplug/plug something and watch which number changes.
# Also shows a socket that sees a device which FAILS to start (kernel "error -32"/"-71").
names() { case "$1" in
    10c4:ea60) echo "lidar (RPLidar CP210x)";; 1a86:55d3) echo "arm servo board";;
    2886:001a) echo "reSpeaker mic array";;   303a:1001) echo "PRIZM base (ESP32)";;
    03e7:*) echo "OAK-D depth camera";;       32e6:9251) echo "wrist camera";;
    046d:c52b) echo "keyboard/mouse receiver";; 2e8a:*) echo "Raspberry Pi USB hub";;
    0484:5750) echo "touchscreen";; *) echo "$1";; esac; }
show() {
    for p in 1 2 3 4; do
        out=""
        for d in /sys/bus/usb/devices/1-2.$p /sys/bus/usb/devices/2-1.$p; do
            [ -f $d/idVendor ] || continue
            id="$(cat $d/idVendor):$(cat $d/idProduct)"; out="$out $(names $id) [$(cat $d/speed) Mbit]"
            for c in $d.*; do [ -f $c/idVendor ] && out="$out  + $(names $(cat $c/idVendor):$(cat $c/idProduct))"; done
        done
        bad=$(journalctl -k --since "-60 s" --no-pager 2>/dev/null | grep -c "usb 1-2.$p: device descriptor read\|usb 1-2.$p: device not accepting")
        [ "$bad" -gt 0 ] && out="$out  !! a device here FAILS to start"
        echo "  socket $p:${out:- (empty)}"
    done
}
if [ "${1:-}" = watch ]; then while true; do clear; date +%T; show; sleep 1; done; else show; fi
