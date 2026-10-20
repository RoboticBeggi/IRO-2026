# ESP32-S3 base motor controller: full flash dump (20 Oct 2026)

Found on USB as 303a:1001 (Espressif USB-Serial/JTAG), MAC 58:8c:81:ea:a3:34.
ESP32-S3 QFN56 rev 0.2, 8 MB flash, 2 MB PSRAM.

- `flash_full.bin`: all 8 MB (sha256 in flash_full.sha256). Restores the board exactly:
  `esptool --port /dev/ttyACM1 write-flash 0 flash_full.bin`
- `app0.bin`: the program (362 KB). Arduino-ESP32 3.3.10, sketch compiled 1 Aug 2026
  on the old Jetson (paths /home/orin_nano/.arduino15/...). No source on the board.
- Partitions: nvs (empty), otadata, app0 (the program), app1 (empty), spiffs (empty), coredump.
- `strings.txt`: every text string in the program.

Recovered command protocol (text lines, one reply each):

| Command | Reply | Probable meaning |
|---|---|---|
| `PING` | `PONG` | link check |
| `STOP` | `OK STOP` | stop all motors |
| `MOVE %d %d %d` | `OK MOVE` | mecanum vx vy wz |
| `RAW %d %d %d %d` | `OK RAW` | power per wheel |
| `MOTOR %d %d` | `OK MOTOR` | one motor, power |
| `ENC` / `ENC 0` | ? / `OK ENC0` | read / reset encoders |
| `WDT %ld` | `OK WDT` | watchdog timeout (ms); prints `WDT STOP` when it fires |
| other | `ERR bad <CMD> format`, `ERR unknown command` | |

This board IS the TETRIX PRIZM Pro (Pitsco library "TETRIX ESP32S3 PRIZM PRO
v1.15"): DC motor chip at I2C 5 and servo chip at 6 on the internal bus
(Wire1, pins 1/2), expansion controllers at 1-4 on Wire (pins 8/9).

Why it was silent on 20 Oct: PRIZM Pro programs call PrizmBegin(), which waits
for the green START button before running (and before enabling motors). Press
START (wheels off the ground), then PING over USB should answer PONG.

Opening the USB port: keep pyserial's defaults (DTR and RTS on). Setting them
off on open passes through RTS-on/DTR-off, which resets the ESP32-S3 (and the
program then waits for START again). Tested 20 Oct: default open = no reset.
