"""Flash-and-profile all 32x32 TFLite models using the VFS image strategy.

This runner avoids REPL file uploads entirely:
  1. Build validation binaries from the 32x32 JPG folder + classes_validacao.txt.
  2. Patch the current model's Softmax scale bytes using the existing helper.
  3. Build a LittleFS image containing:
       /model.tflite
       /profile_device.py
       /images_32.bin
       /labels_32.bin
  4. Flash the image to the VFS partition at 0x200000 with no backup.
  5. Drive profile_device.py over COM12 to find min arena, profile, and evaluate.

Results:
  vfs_32_profile_results.json
  vfs_32_profile_results.csv
  vfs_32_logs/<model>.log
"""
import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import serial
from PIL import Image

import _tflite_meta
import flash_vfs
import vfs_common as S


MODELS_DIR = Path(__file__).resolve().parent
OUT_DIR = MODELS_DIR / "tflite_out"
DEVICE_SCRIPT = MODELS_DIR / "profile_device.py"
STAGE_DIR = MODELS_DIR / "_vfs_stage_32"
LOG_DIR = MODELS_DIR / "vfs_32_logs"
RESULTS_JSON = MODELS_DIR / "vfs_32_profile_results.json"
RESULTS_CSV = MODELS_DIR / "vfs_32_profile_results.csv"

VALIDATION_ROOT = Path(
    MODELS_DIR / "data" / "validation_images"
    / "validacao_indoor_32_3-20260428T030919Z-3-001"
    / "validacao_indoor_32_3"
)
VALIDATION_TXT = VALIDATION_ROOT / "classes_validacao.txt"

PORT = "COM12"
FLASH_BAUD = 460800
REPL_BAUD = 115200
N_IMAGES = 200
IMAGE_W = 32
IMAGE_H = 32
IMAGE_C = 3
IMAGE_STRIDE = IMAGE_W * IMAGE_H * IMAGE_C


def parse_model_name(name):
    match = re.search(r"_a([0-9.]+)_rep(\d+)_(\d+)_", name)
    if not match:
        return {"alpha": "", "rep": "", "input_size": ""}
    return {
        "alpha": match.group(1),
        "rep": int(match.group(2)),
        "input_size": int(match.group(3)),
    }


def discover_32_models():
    models = []
    for path in sorted(OUT_DIR.glob("*.tflite"), key=lambda p: p.name):
        parsed = parse_model_name(path.name)
        if parsed["input_size"] == 32:
            models.append(path)
    if not models:
        raise SystemExit("No 32x32 models found in %s" % OUT_DIR)
    return models


def read_validation_rows():
    if not VALIDATION_TXT.exists():
        raise SystemExit("Missing validation TXT: %s" % VALIDATION_TXT)
    rows = []
    for line_no, line in enumerate(VALIDATION_TXT.read_text().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 3:
            raise SystemExit("Bad validation line %d: %r" % (line_no, line))
        name, label_y, label_x = parts
        img_path = VALIDATION_ROOT / name
        if not img_path.exists():
            raise SystemExit("TXT references missing image: %s" % img_path)
        rows.append((name, int(label_y), int(label_x), img_path))
    if len(rows) != N_IMAGES:
        raise SystemExit("Expected %d validation rows, got %d" % (N_IMAGES, len(rows)))
    return rows


def prepare_validation_bins(force=False):
    STAGE_DIR.mkdir(exist_ok=True)
    images_bin = STAGE_DIR / "images_32.bin"
    labels_bin = STAGE_DIR / "labels_32.bin"
    manifest = STAGE_DIR / "validation_32_manifest.csv"
    expected_img_bytes = N_IMAGES * IMAGE_STRIDE
    expected_label_bytes = N_IMAGES * 2

    if (
        not force
        and images_bin.exists()
        and images_bin.stat().st_size == expected_img_bytes
        and labels_bin.exists()
        and labels_bin.stat().st_size == expected_label_bytes
        and manifest.exists()
    ):
        return images_bin, labels_bin, manifest

    rows = read_validation_rows()
    image_bytes = bytearray()
    label_y = bytearray()
    label_x = bytearray()

    with manifest.open("w", newline="", encoding="utf-8") as mf:
        writer = csv.DictWriter(mf, fieldnames=["idx", "filename", "label_y", "label_x"])
        writer.writeheader()
        for idx, (name, y, x, img_path) in enumerate(rows):
            img = Image.open(img_path).convert("RGB")
            if img.size != (IMAGE_W, IMAGE_H):
                raise SystemExit("%s is %s, expected 32x32" % (img_path, img.size))
            image_bytes.extend(img.tobytes())
            label_y.append(y & 0xFF)
            label_x.append(x & 0xFF)
            writer.writerow({"idx": idx, "filename": name, "label_y": y, "label_x": x})

    images_bin.write_bytes(bytes(image_bytes))
    labels_bin.write_bytes(bytes(label_y + label_x))
    return images_bin, labels_bin, manifest


def patched_model_file(model_path):
    STAGE_DIR.mkdir(exist_ok=True)
    data = bytearray(model_path.read_bytes())
    patches = S.R.patch_softmax_scales(data)
    patched = STAGE_DIR / "model.tflite"
    patched.write_bytes(bytes(data))
    return patched, patches


def flash_job(model_path, images_bin, labels_bin, port, baud):
    patched_model, patches = patched_model_file(model_path)
    image_path = MODELS_DIR / "_vfs_new.bin"
    files = [
        (patched_model, "model.tflite"),
        (DEVICE_SCRIPT, "profile_device.py"),
        (images_bin, "images_32.bin"),
        (labels_bin, "labels_32.bin"),
    ]

    print("  building LittleFS image for %s" % model_path.name, flush=True)
    flash_vfs.build_image(str(image_path), files)

    cmd = [
        sys.executable,
        "-m",
        "esptool",
        "--chip",
        "esp32s3",
        "--port",
        port,
        "--baud",
        str(baud),
        "--before",
        "default-reset",
        "--after",
        "hard-reset",
        "write-flash",
        hex(flash_vfs.VFS_OFFSET),
        str(image_path),
    ]
    print("  flashing VFS without backup", flush=True)
    print("  $ " + " ".join(cmd), flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.time() - t0
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise RuntimeError("esptool write-flash failed for %s" % model_path.name)
    print("  flash wall time %.1fs" % dt, flush=True)
    return patches, dt


def run_eval(ser, arena, stride, log_path):
    S.R.reset_and_repl(ser)
    code = (
        "import profile_device; "
        "profile_device.evaluate(%d, 'images_32.bin', 'labels_32.bin', %d, %d)"
        % (arena, stride, N_IMAGES)
    )
    with log_path.open("w", encoding="utf-8") as log:
        text = S.repl_call(ser, code, "EVAL_JSON:", timeout=1800, log_file=log)
    data = S.json_from_output(text, "EVAL_JSON:")
    if data is None:
        return {"status": "timeout", "tail": text[-1200:]}
    return data


def write_csv(rows):
    if not rows:
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with RESULTS_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_existing(resume):
    if not resume or not RESULTS_JSON.exists():
        return []
    try:
        loaded = json.loads(RESULTS_JSON.read_text())
    except ValueError:
        return []
    return loaded if isinstance(loaded, list) else []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=PORT)
    parser.add_argument("--flash-baud", type=int, default=FLASH_BAUD)
    parser.add_argument("--repl-baud", type=int, default=REPL_BAUD)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force-validation-bins", action="store_true")
    parser.add_argument("--model", action="append",
                        help="Run only models whose filename contains this text. May repeat.")
    args = parser.parse_args()

    LOG_DIR.mkdir(exist_ok=True)
    images_bin, labels_bin, manifest = prepare_validation_bins(args.force_validation_bins)
    models = discover_32_models()
    if args.model:
        needles = tuple(args.model)
        models = [m for m in models if any(n in m.name for n in needles)]
    if not models:
        raise SystemExit("No selected 32x32 models to run.")

    rows = load_existing(args.resume)
    completed = {row.get("name") for row in rows if row.get("status") == "ok"}
    fw = S.firmware_sizes()

    print("Validation TXT: %s" % VALIDATION_TXT)
    print("Validation manifest: %s" % manifest)
    print("Images bin: %s (%d B)" % (images_bin, images_bin.stat().st_size))
    print("Labels bin: %s (%d B)" % (labels_bin, labels_bin.stat().st_size))
    print("Models: %s" % ", ".join(m.name for m in models))

    for index, model in enumerate(models, 1):
        print("\n[%d/%d] %s" % (index, len(models), model.name), flush=True)
        if args.resume and model.name in completed:
            print("  skip existing ok result", flush=True)
            continue

        meta = _tflite_meta.meta(model)
        stride = S.input_stride(meta)
        if stride != IMAGE_STRIDE:
            print("  skip: input stride %d != %d" % (stride, IMAGE_STRIDE), flush=True)
            continue

        patches, flash_seconds = flash_job(model, images_bin, labels_bin,
                                           args.port, args.flash_baud)
        time.sleep(1.0)

        with serial.Serial(args.port, args.repl_baud, timeout=0.05) as ser:
            min_arena, history = S.find_min_arena(ser, stride, meta["file_bytes"])
            row = S.model_row_base(model, meta, patches)
            parsed = parse_model_name(model.name)
            row["input_size"] = parsed["input_size"]
            row["validation_txt"] = str(VALIDATION_TXT)
            row["validation_manifest"] = str(manifest)
            row["validation_image_count"] = N_IMAGES
            row["flash_seconds"] = round(flash_seconds, 3)
            row["arena_resolution_bytes"] = S.ARENA_RESOLUTION
            row["arena_history"] = history

            if min_arena is None:
                row["status"] = "arena_search_failed"
                rows.append(row)
                RESULTS_JSON.write_text(json.dumps(rows, indent=2))
                write_csv(rows)
                continue

            row["min_arena_bytes"] = min_arena
            print("  min arena = %d" % min_arena, flush=True)

            profile_arena = min_arena
            profile = S.run_profile(ser, profile_arena, stride)
            if profile.get("status") != "ok":
                profile_arena = max(min_arena * 2, min_arena + 50_000)
                print("  profile at min failed; retry arena = %d" % profile_arena)
                profile = S.run_profile(ser, profile_arena, stride)
            row["profile_arena_bytes"] = profile_arena
            row["profile_status"] = profile.get("status")
            S.flatten_runtime("profile_", profile, row)

            eval_arena = min_arena
            log_path = LOG_DIR / ("%s.log" % model.stem)
            eval_data = run_eval(ser, eval_arena, stride, log_path)
            if eval_data.get("status") != "ok":
                eval_arena = max(min_arena * 2, min_arena + 50_000)
                print("  eval at min failed; retry arena = %d" % eval_arena)
                eval_data = run_eval(ser, eval_arena, stride, log_path)
            row["eval_arena_bytes"] = eval_arena
            row["eval_status"] = eval_data.get("status")
            row["eval_log"] = str(log_path)
            S.flatten_runtime("eval_", eval_data, row)
            for key in ("eval_count", "correct_y", "correct_x", "accy", "accx"):
                row[key] = eval_data.get(key)

            row["status"] = (
                "ok"
                if profile.get("status") == "ok" and eval_data.get("status") == "ok"
                else "failed"
            )
            S.add_flash_totals(row, eval_data if eval_data.get("status") == "ok" else profile, fw)

            rows = [r for r in rows if r.get("name") != model.name]
            rows.append(row)
            RESULTS_JSON.write_text(json.dumps(rows, indent=2))
            write_csv(rows)
            print(
                "  result status=%s accy=%s accx=%s avg_us=%s"
                % (row["status"], row.get("accy"), row.get("accx"),
                   row.get("eval_avg_invoke_us")),
                flush=True,
            )

    RESULTS_JSON.write_text(json.dumps(rows, indent=2))
    write_csv(rows)
    print("\nWROTE %s" % RESULTS_JSON)
    print("WROTE %s" % RESULTS_CSV)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
