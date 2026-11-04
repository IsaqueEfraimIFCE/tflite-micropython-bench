"""Validate converted numbered 32x32 MobileNet models on MicroPython via VFS.

Uses the established direct LittleFS strategy:
  - stage the 200 validation images and labels as binary files,
  - stage one model as /model.tflite,
  - build a LittleFS image,
  - flash it directly to the MicroPython VFS partition at 0x200000,
  - run profile_device.evaluate() over the COM12 REPL.

The source models are the local uint8 TFLM-compatible conversions in:
  models/data/tflm_uint8
"""

from __future__ import annotations

import argparse
import csv
import json
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
SOURCE_DIR = MODELS_DIR / "data" / "tflm_uint8"
DEVICE_SCRIPT = MODELS_DIR / "profile_device.py"
STAGE_DIR = MODELS_DIR / "_vfs_stage_32_numbered"
LOG_DIR = MODELS_DIR / "vfs_32_numbered_logs"
RESULTS_JSON = MODELS_DIR / "vfs_32_numbered_results.json"
RESULTS_CSV = MODELS_DIR / "vfs_32_numbered_results.csv"
SUMMARY_CSV = MODELS_DIR / "vfs_32_numbered_summary.csv"

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
MODEL_RE = re.compile(r"MobileNetV1\[(?P<num>\d+)\]_uint8\.tflite$")


def alpha_for_model(model_number: int) -> float:
    if 80 <= model_number <= 83:
        return 0.25
    if 84 <= model_number <= 87:
        return 0.5
    if 88 <= model_number <= 91:
        return 1.0
    if 92 <= model_number <= 95:
        return 0.25
    if 96 <= model_number <= 99:
        return 0.5
    if 100 <= model_number <= 103:
        return 1.0
    if 104 <= model_number <= 107:
        return 0.25
    if 108 <= model_number <= 111:
        return 0.5
    if 112 <= model_number <= 115:
        return 1.0
    if 116 <= model_number <= 119:
        return 0.25
    if 120 <= model_number <= 123:
        return 0.5
    if 124 <= model_number <= 127:
        return 1.0
    raise ValueError("Unknown MobileNetV1 model number: %d" % model_number)


def resolution_for_model(model_number: int) -> int:
    if 80 <= model_number <= 91:
        return 128
    if 92 <= model_number <= 103:
        return 64
    if 104 <= model_number <= 115:
        return 32
    if 116 <= model_number <= 127:
        return 224
    raise ValueError("Unknown MobileNetV1 model number: %d" % model_number)


def model_number(path: Path) -> int:
    match = MODEL_RE.fullmatch(path.name)
    if not match:
        raise ValueError("Unexpected model filename: %s" % path.name)
    return int(match.group("num"))


def discover_models(selected: list[str] | None) -> list[Path]:
    models = []
    for path in sorted(SOURCE_DIR.glob("*_uint8.tflite"), key=lambda p: p.name):
        if not MODEL_RE.fullmatch(path.name):
            continue
        number = model_number(path)
        if resolution_for_model(number) != 32:
            continue
        if selected and not any(token in path.name or token == str(number) for token in selected):
            continue
        meta = _tflite_meta.meta(path)
        if S.input_stride(meta) != IMAGE_STRIDE:
            continue
        models.append(path)
    if not models:
        raise SystemExit("No converted 32x32 numbered models found in %s" % SOURCE_DIR)
    return models


def read_validation_rows():
    if not VALIDATION_TXT.exists():
        raise SystemExit("Missing validation TXT: %s" % VALIDATION_TXT)
    rows = []
    for line_no, line in enumerate(VALIDATION_TXT.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 3:
            raise SystemExit("Bad validation line %d: %r" % (line_no, line))
        name, label_y, label_x = parts
        image_path = VALIDATION_ROOT / name
        if not image_path.exists():
            raise SystemExit("Missing validation image: %s" % image_path)
        rows.append((name, int(label_y), int(label_x), image_path))
    if len(rows) != N_IMAGES:
        raise SystemExit("Expected %d validation rows, got %d" % (N_IMAGES, len(rows)))
    return rows


def prepare_validation_bins(force: bool):
    STAGE_DIR.mkdir(exist_ok=True)
    images_bin = STAGE_DIR / "images_32.bin"
    labels_bin = STAGE_DIR / "labels_32.bin"
    manifest = STAGE_DIR / "validation_32_manifest.csv"
    expected_image_bytes = N_IMAGES * IMAGE_STRIDE
    expected_label_bytes = N_IMAGES * 2

    if (
        not force
        and images_bin.exists()
        and images_bin.stat().st_size == expected_image_bytes
        and labels_bin.exists()
        and labels_bin.stat().st_size == expected_label_bytes
        and manifest.exists()
    ):
        return images_bin, labels_bin, manifest

    image_bytes = bytearray()
    label_y = bytearray()
    label_x = bytearray()
    rows = read_validation_rows()
    with manifest.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["idx", "filename", "label_y", "label_x"])
        writer.writeheader()
        for idx, (name, y, x, image_path) in enumerate(rows):
            image = Image.open(image_path).convert("RGB")
            if image.size != (IMAGE_W, IMAGE_H):
                raise SystemExit("%s is %s, expected 32x32" % (image_path, image.size))
            image_bytes.extend(image.tobytes())
            label_y.append(y & 0xFF)
            label_x.append(x & 0xFF)
            writer.writerow({"idx": idx, "filename": name, "label_y": y, "label_x": x})

    images_bin.write_bytes(bytes(image_bytes))
    labels_bin.write_bytes(bytes(label_y + label_x))
    return images_bin, labels_bin, manifest


def patched_model_file(model_path: Path):
    data = bytearray(model_path.read_bytes())
    patches = S.R.patch_softmax_scales(data)
    staged = STAGE_DIR / "model.tflite"
    staged.write_bytes(bytes(data))
    return staged, patches


def flash_vfs_image(model_path: Path, images_bin: Path, labels_bin: Path, port: str, baud: int):
    staged_model, patches = patched_model_file(model_path)
    image_path = MODELS_DIR / "_vfs_new.bin"
    files = [
        (staged_model, "model.tflite"),
        (DEVICE_SCRIPT, "profile_device.py"),
        (images_bin, "images_32.bin"),
        (labels_bin, "labels_32.bin"),
    ]

    print("  building LittleFS image", flush=True)
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
    started = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    seconds = time.time() - started
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise RuntimeError("esptool write-flash failed for %s" % model_path.name)
    print("  flash_seconds=%.3f" % seconds, flush=True)
    return patches, seconds


def run_eval(ser, arena: int, stride: int, log_path: Path):
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


def write_tables(rows: list[dict]):
    RESULTS_JSON.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    if not rows:
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with RESULTS_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    summary_fields = [
        "model_number",
        "model",
        "alpha",
        "resolution",
        "status",
        "eval_count",
        "correct_y",
        "correct_x",
        "accy",
        "accx",
        "min_arena_bytes",
        "eval_arena_bytes",
        "profile_arena_bytes",
        "eval_avg_invoke_us",
        "model_file_bytes",
        "eval_ram_model_bytes",
        "eval_ram_arena_bytes",
        "eval_ram_total_bytes",
        "eval_vfs_used_bytes",
        "full_flash_used_bytes",
    ]
    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=summary_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: int(r.get("model_number", 0))))


def load_existing(resume: bool):
    if not resume or not RESULTS_JSON.exists():
        return []
    try:
        rows = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
    except ValueError:
        return []
    return rows if isinstance(rows, list) else []


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=PORT)
    parser.add_argument("--flash-baud", type=int, default=FLASH_BAUD)
    parser.add_argument("--repl-baud", type=int, default=REPL_BAUD)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force-validation-bins", action="store_true")
    parser.add_argument("--model", action="append", help="Model number or filename substring.")
    args = parser.parse_args()

    LOG_DIR.mkdir(exist_ok=True)
    STAGE_DIR.mkdir(exist_ok=True)
    images_bin, labels_bin, manifest = prepare_validation_bins(args.force_validation_bins)
    models = discover_models(args.model)
    rows = load_existing(args.resume)
    completed = {row.get("model") for row in rows if row.get("status") == "ok"}
    firmware = S.firmware_sizes()

    print("Source models: %s" % SOURCE_DIR)
    print("Validation TXT: %s" % VALIDATION_TXT)
    print("Validation manifest: %s" % manifest)
    print("Images bin: %s (%d B)" % (images_bin, images_bin.stat().st_size))
    print("Labels bin: %s (%d B)" % (labels_bin, labels_bin.stat().st_size))
    print("Selected models: %s" % ", ".join(model.name for model in models))

    for index, model_path in enumerate(models, 1):
        number = model_number(model_path)
        print("\n[%d/%d] %s alpha=%s" % (index, len(models), model_path.name, alpha_for_model(number)), flush=True)
        if args.resume and model_path.name in completed:
            print("  skip existing ok result", flush=True)
            continue

        meta = _tflite_meta.meta(model_path)
        stride = S.input_stride(meta)
        patches, flash_seconds = flash_vfs_image(
            model_path, images_bin, labels_bin, args.port, args.flash_baud
        )
        time.sleep(1.0)

        with serial.Serial(args.port, args.repl_baud, timeout=0.05) as ser:
            min_arena, history = S.find_min_arena(ser, stride, meta["file_bytes"])
            row = S.model_row_base(model_path, meta, patches)
            row["model"] = model_path.name
            row["model_number"] = number
            row["alpha"] = alpha_for_model(number)
            row["resolution"] = resolution_for_model(number)
            row["input_size"] = resolution_for_model(number)
            row["validation_txt"] = str(VALIDATION_TXT)
            row["validation_manifest"] = str(manifest)
            row["validation_image_count"] = N_IMAGES
            row["flash_seconds"] = round(flash_seconds, 3)
            row["arena_resolution_bytes"] = S.ARENA_RESOLUTION
            row["arena_history"] = history
            row["preproc"] = "raw_uint8_rgb_nhwc"
            row["mapping"] = "out0->accx,out1->accy"

            if min_arena is None:
                row["status"] = "arena_search_failed"
                rows = [existing for existing in rows if existing.get("model") != model_path.name]
                rows.append(row)
                write_tables(rows)
                continue

            row["min_arena_bytes"] = min_arena
            print("  min_arena=%d" % min_arena, flush=True)

            profile_arena = min_arena
            profile = S.run_profile(ser, profile_arena, stride)
            if profile.get("status") != "ok":
                profile_arena = max(min_arena * 2, min_arena + 50_000)
                print("  profile retry arena=%d" % profile_arena, flush=True)
                profile = S.run_profile(ser, profile_arena, stride)
            row["profile_arena_bytes"] = profile_arena
            row["profile_status"] = profile.get("status")
            S.flatten_runtime("profile_", profile, row)

            eval_arena = min_arena
            log_path = LOG_DIR / ("%s.log" % model_path.stem)
            eval_data = run_eval(ser, eval_arena, stride, log_path)
            if eval_data.get("status") != "ok":
                eval_arena = max(min_arena * 2, min_arena + 50_000)
                print("  eval retry arena=%d" % eval_arena, flush=True)
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
            S.add_flash_totals(
                row,
                eval_data if eval_data.get("status") == "ok" else profile,
                firmware,
            )

        rows = [existing for existing in rows if existing.get("model") != model_path.name]
        rows.append(row)
        write_tables(rows)
        print(
            "  result=%s accy=%s accx=%s avg_us=%s"
            % (row["status"], row.get("accy"), row.get("accx"), row.get("eval_avg_invoke_us")),
            flush=True,
        )

    write_tables(rows)
    print("\nWROTE %s" % RESULTS_JSON)
    print("WROTE %s" % RESULTS_CSV)
    print("WROTE %s" % SUMMARY_CSV)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
