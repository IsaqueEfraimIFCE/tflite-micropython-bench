"""Drive the deep-sleep baseline: quiet everything, both cores off, capture the
RTC-only floor current, then detect the reboot on timer wake.

Assumes MicroPython + power_probe.py are already flashed to VFS.

Usage:
  python run_baseline.py                    # 10 s deep-sleep window
  python run_baseline.py --sleep-ms 20000 --marker-pin 21
"""
import argparse
import json
import time

import serial

import run_patched_tflite_bench as R
import vfs_common as S


PORT = "COM12"
REPL_BAUD = 115200


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default=PORT)
    ap.add_argument("--baud", type=int, default=REPL_BAUD)
    ap.add_argument("--sleep-ms", type=int, default=10000)
    ap.add_argument("--marker-pin", type=int, default=21)
    args = ap.parse_args()

    with serial.Serial(args.port, args.baud, timeout=0.05) as ser:
        R.reset_and_repl(ser)

        code = ("import power_probe; power_probe.deep_sleep_baseline("
                "sleep_ms=%d, marker_pin=%d)" % (args.sleep_ms, args.marker_pin))
        R._seq[0] += 1
        ser.write(code.encode("utf-8") + b"\r\n")

        # 1) Catch the pre-sleep BASELINE_JSON marker.
        pre = b""
        end = time.time() + 10
        while time.time() < end and b"BASELINE_JSON:" not in pre:
            pre += ser.read(4096) or b""
            time.sleep(0.01)
        data = S.json_from_output(pre.decode("utf-8", "replace"), "BASELINE_JSON:")
        if data is None:
            print("ERROR: no BASELINE_JSON before sleep. Tail:")
            print(repr(pre[-400:]))
            return 1
        print("ENTERED DEEP SLEEP:", json.dumps(data))
        t_sleep = time.time()
        print(">>> analyzer window OPEN — floor current for ~%.1f s <<<"
              % (args.sleep_ms / 1000.0))

        # 2) Hold through the sleep window and watch for the reboot banner.
        boot = b""
        end = time.time() + args.sleep_ms / 1000.0 + 12
        while time.time() < end:
            boot += ser.read(4096) or b""
            if b"MicroPython" in boot or b"rst:0x" in boot:
                break
            time.sleep(0.02)
        woke_after = time.time() - t_sleep

        if b"MicroPython" in boot or b"rst:0x" in boot:
            print("WOKE / REBOOTED after %.2f s (requested %.2f s)"
                  % (woke_after, args.sleep_ms / 1000.0))
            print(">>> analyzer window CLOSED — measure avg current in the flat"
                  " region between ENTER and WOKE <<<")
            print("BOOT TAIL:", repr(boot[-200:]))
            return 0

        print("WARNING: no reboot detected within window. Tail:")
        print(repr(boot[-400:]))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
