"""Sweep CPU frequency (and optionally arena) for one model and log each
power_probe run to CSV. Pairs with an external current analyzer triggered on
the marker GPIO: each row records the invoke() timing + quiet-down state so you
can correlate latency vs. clock against the captured current trace.

Assumes the board already has MicroPython + the target model.tflite +
power_probe.py flashed to VFS (see flash_vfs.py). This script only drives the
REPL; it does not flash.

Usage:
  python run_power_sweep.py --stride 3072 --arena 150000 \
      --cpu-mhz 240 160 80 --iters 200 --idle-ms 2000 --marker-pin 21
  python run_power_sweep.py --stride 3072 --arena 120000 150000 200000 --cpu-mhz 240
"""
import argparse
import csv
import json
import time
from pathlib import Path

import serial

import run_patched_tflite_bench as R
import vfs_common as S


MODELS_DIR = Path(__file__).resolve().parent
RESULTS_CSV = MODELS_DIR / "power_sweep_results.csv"

PORT = "COM12"
REPL_BAUD = 115200

CSV_FIELDS = [
    "timestamp", "cpu_mhz", "cpu_hz_actual", "arena", "image_stride", "iters",
    "status", "avg_invoke_us", "min_invoke_us", "max_invoke_us",
    "marker_pin", "idle_ms", "file_bytes", "wifi", "ble", "led", "error",
]


def one_run(ser, arena, stride, cpu_hz, iters, marker_pin, idle_ms, timeout):
    R.reset_and_repl(ser)
    code = (
        "import power_probe; power_probe.run(%d, %d, iters=%d, marker_pin=%d, "
        "idle_ms=%d, cpu_hz=%d)"
        % (arena, stride, iters, marker_pin, idle_ms, cpu_hz)
    )
    text = S.repl_call(ser, code, "POWER_JSON:", timeout=timeout)
    data = S.json_from_output(text, "POWER_JSON:")
    if data is None:
        return {"status": "timeout", "tail": text[-400:]}
    return data


def to_row(data, cpu_mhz):
    quiet = data.get("quiet") or {}
    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cpu_mhz": cpu_mhz,
        "cpu_hz_actual": quiet.get("cpu_hz"),
        "arena": data.get("arena"),
        "image_stride": data.get("image_stride"),
        "iters": data.get("iters"),
        "status": data.get("status"),
        "avg_invoke_us": data.get("avg_invoke_us"),
        "min_invoke_us": data.get("min_invoke_us"),
        "max_invoke_us": data.get("max_invoke_us"),
        "marker_pin": data.get("marker_pin"),
        "idle_ms": data.get("idle_ms"),
        "file_bytes": data.get("file_bytes"),
        "wifi": quiet.get("wifi"),
        "ble": quiet.get("ble"),
        "led": quiet.get("led"),
        "error": data.get("error"),
    }


def write_csv(rows):
    with RESULTS_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default=PORT)
    ap.add_argument("--baud", type=int, default=REPL_BAUD)
    ap.add_argument("--stride", type=int, required=True,
                    help="input stride = W*H*C (e.g. 3072 for 32x32x3)")
    ap.add_argument("--arena", type=int, nargs="+", default=[150000],
                    help="one or more arena sizes to sweep")
    ap.add_argument("--cpu-mhz", type=int, nargs="+", default=[240, 160, 80],
                    help="one or more CPU frequencies in MHz to sweep")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--idle-ms", type=int, default=2000)
    ap.add_argument("--marker-pin", type=int, default=21)
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--settle-s", type=float, default=3.0,
                    help="pause between runs to let the analyzer separate windows")
    args = ap.parse_args()

    rows = []
    with serial.Serial(args.port, args.baud, timeout=0.05) as ser:
        for arena in args.arena:
            for mhz in args.cpu_mhz:
                print("RUN arena=%d cpu=%dMHz ..." % (arena, mhz), flush=True)
                data = one_run(ser, arena, args.stride, mhz * 1_000_000,
                               args.iters, args.marker_pin, args.idle_ms,
                               args.timeout)
                row = to_row(data, mhz)
                rows.append(row)
                write_csv(rows)
                print("  status=%s avg_us=%s min=%s max=%s cpu_hz=%s"
                      % (row["status"], row["avg_invoke_us"],
                         row["min_invoke_us"], row["max_invoke_us"],
                         row["cpu_hz_actual"]), flush=True)
                if data.get("status") != "ok":
                    print("  RAW: %s" % json.dumps(data)[:400], flush=True)
                time.sleep(args.settle_s)

    print("\nWROTE %s (%d rows)" % (RESULTS_CSV, len(rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
