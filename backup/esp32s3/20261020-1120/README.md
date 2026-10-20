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

Uses the I2C master (likely TETRIX motor expansion controllers). The commands
did NOT answer over USB, nor on the Jetson's /dev/ttyTHS1-2: the command line
is probably the board's UART pins, wired to something not connected now.
