"""Accuracy-only MicroPython validation for all converted numbered models.

This uses direct LittleFS VFS flashing on COM12. It intentionally does not
profile memory, flash use, or inference time. The only final metrics are ACCY
and ACCX over the 200 validation images for the model's own resolution.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
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
DEVICE_SCRIPT = MODELS_DIR / "accuracy_device.py"
STAGE_DIR = MODELS_DIR / "_vfs_stage_accuracy"
LOG_DIR = MODELS_DIR / "vfs_accuracy_logs"
RESULTS_JSON = MODELS_DIR / "vfs_accuracy_all_results.json"
SUMMARY_CSV = MODELS_DIR / "vfs_accuracy_all_summary.csv"
CHUNKS_CSV = MODELS_DIR / "vfs_accuracy_all_chunks.csv"
RUN_STATE_JSON = MODELS_DIR / "vfs_accuracy_all_state.json"

PORT = "COM12"
FLASH_BAUD = 460800
REPL_BAUD = 115200
N_IMAGES = 200
MODEL_RE = re.compile(r"MobileNetV1\[(?P<num>\d+)\]_uint8\.tflite$")

# Keep a margin under the raw 6 MB LittleFS image because metadata consumes
# blocks too. The runner reduces chunk size further if image construction fails.
SAFE_PAYLOAD_BYTES = 5_450_000


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
    raise ValueError("Unknown model number: %d" % model_number)


def resolution_for_model(model_number: int) -> int:
    if 80 <= model_number <= 91:
        return 128
    if 92 <= model_number <= 103:
        return 64
    if 104 <= model_number <= 115:
        return 32
    if 116 <= model_number <= 127:
        return 224
    raise ValueError("Unknown model number: %d" % model_number)


def model_number(path: Path) -> int:
    match = MODEL_RE.fullmatch(path.name)
    if not match:
        raise ValueError("Unexpected model filename: %s" % path.name)
    return int(match.group("num"))


def validation_root(size: int) -> Path:
    roots = sorted(
        (MODELS_DIR / "data" / "validation_images").glob(
            "validacao_indoor_%d_3-*" % size
        )
    )
    if not roots:
        raise FileNotFoundError("No validation folder for %dx%d" % (size, size))
    nested = roots[0] / ("validacao_indoor_%d_3" % size)
    if not nested.exists():
        raise FileNotFoundError("No nested validation folder: %s" % nested)
    return nested


def discover_models(selected: list[str] | None) -> list[Path]:
    models = []
    for path in sorted(SOURCE_DIR.glob("*_uint8.tflite"), key=lambda p: p.name):
        if not MODEL_RE.fullmatch(path.name):
            continue
        number = model_number(path)
        if selected and not any(token in path.name or token == str(number) for token in selected):
            continue
        models.append(path)
    if not models:
        raise SystemExit("No converted models selected in %s" % SOURCE_DIR)

    return sorted(
        models,
        key=lambda p: (
            resolution_for_model(model_number(p)),
            alpha_for_model(model_number(p)),
            model_number(p),
        ),
    )


def read_validation_rows(size: int):
    root = validation_root(size)
    txt = root / "classes_validacao.txt"
    rows = []
    for line_no, line in enumerate(txt.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 3:
            raise SystemExit("Bad validation line %d in %s: %r" % (line_no, txt, line))
        name, label_y, label_x = parts
        image_path = root / name
        if not image_path.exists():
            raise SystemExit("Missing image referenced by %s: %s" % (txt, image_path))
        rows.append((name, int(label_y), int(label_x), image_path))
    if len(rows) != N_IMAGES:
        raise SystemExit("Expected %d rows in %s, got %d" % (N_IMAGES, txt, len(rows)))
    return root, txt, rows


def arena_for(alpha: float, size: int) -> int:
    # Fixed safe arenas derived from previous successful MicroPython runs for
    # the same MobileNetV1 architecture classes. No arena search is performed.
    floors = {
        (0.25, 32): 35742,
        (0.5, 32): 61522,
        (1.0, 32): 117577,
        (0.25, 64): 60351,
        (0.5, 64): 98510,
        (1.0, 64): 191406,
        (0.25, 128): 158496,
        (0.5, 128): 246093,
        (1.0, 128): 486327,
        (0.25, 224): 428906,
        (0.5, 224): 651488,
        (1.0, 224): 1297265,
    }
    floor = floors[(alpha, size)]
    return max(int(math.ceil(floor * 1.6)), floor + 50_000)


def write_validation_chunk(size: int, chunk_rows, chunk_index: int):
    image_stride = size * size * 3
    images_path = STAGE_DIR / "images.bin"
    labels_path = STAGE_DIR / "labels.bin"
    manifest_path = STAGE_DIR / ("manifest_%d.csv" % chunk_index)

    image_bytes = bytearray()
    label_y = bytearray()
    label_x = bytearray()
    with manifest_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["chunk_local_idx", "global_idx", "filename", "label_y", "label_x"],
        )
        writer.writeheader()
        for local_idx, (global_idx, name, y, x, image_path) in enumerate(chunk_rows):
            image = Image.open(image_path).convert("RGB")
            if image.size != (size, size):
                raise SystemExit("%s is %s, expected %dx%d" % (image_path, image.size, size, size))
            image_bytes.extend(image.tobytes())
            label_y.append(y & 0xFF)
            label_x.append(x & 0xFF)
            writer.writerow({
                "chunk_local_idx": local_idx,
                "global_idx": global_idx,
                "filename": name,
                "label_y": y,
                "label_x": x,
            })

    images_path.write_bytes(bytes(image_bytes))
    labels_path.write_bytes(bytes(label_y + label_x))
    if images_path.stat().st_size != len(chunk_rows) * image_stride:
        raise RuntimeError("Bad image chunk size")
    return images_path, labels_path, manifest_path


def initial_chunk_size(model_bytes: int, size: int) -> int:
    per_image = size * size * 3 + 2
    fixed = DEVICE_SCRIPT.stat().st_size + model_bytes + 80_000
    count = max(1, (SAFE_PAYLOAD_BYTES - fixed) // per_image)
    return min(N_IMAGES, int(count))


def make_chunks(rows, chunk_size: int):
    indexed = [(idx, *row) for idx, row in enumerate(rows)]
    return [indexed[i:i + chunk_size] for i in range(0, len(indexed), chunk_size)]


def stage_model(model_path: Path):
    data = bytearray(model_path.read_bytes())
    patches = S.R.patch_softmax_scales(data)
    staged = STAGE_DIR / "model.tflite"
    staged.write_bytes(bytes(data))
    return staged, patches


def try_build_image(staged_model: Path, images_path: Path, labels_path: Path) -> bool:
    files = [
        (staged_model, "model.tflite"),
        (DEVICE_SCRIPT, "accuracy_device.py"),
        (images_path, "images.bin"),
        (labels_path, "labels.bin"),
    ]
    try:
        flash_vfs.build_image(str(MODELS_DIR / "_vfs_new.bin"), files)
        return True
    except Exception as exc:
        print("  LittleFS build failed: %r" % (exc,), flush=True)
        return False


def flash_current_image(port: str, baud: int):
    image_path = MODELS_DIR / "_vfs_new.bin"
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
    started = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    seconds = time.time() - started
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise RuntimeError("esptool write-flash failed")
    return seconds


def run_chunk_eval(port: str, baud: int, arena: int, stride: int, n_images: int, log_path: Path):
    with serial.Serial(port, baud, timeout=0.05) as ser:
        S.R.reset_and_repl(ser)
        code = (
            "import accuracy_device; "
            "accuracy_device.evaluate(%d, %d, %d, 'images.bin', 'labels.bin')"
            % (arena, stride, n_images)
        )
        with log_path.open("w", encoding="utf-8") as log:
            text = S.repl_call(ser, code, "ACC_JSON:", timeout=7200, log_file=log)
    data = S.json_from_output(text, "ACC_JSON:")
    if data is None:
        return {"status": "timeout", "count": n_images, "correct_y": 0, "correct_x": 0, "tail": text[-1200:]}
    return data


def load_state(path: Path):
    if not path.exists():
        return {"models": [], "chunks": []}
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return {"models": [], "chunks": []}
    if isinstance(state, dict):
        return {
            "models": state.get("models", []),
            "chunks": state.get("chunks", []),
        }
    return {"models": [], "chunks": []}


def write_outputs(model_rows: list[dict], chunk_rows: list[dict]):
    RESULTS_JSON.write_text(json.dumps({"models": model_rows, "chunks": chunk_rows}, indent=2), encoding="utf-8")

    model_fields = [
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
        "chunks_done",
        "chunks_total",
        "arena_bytes",
    ]
    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=model_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(model_rows, key=lambda r: (r["resolution"], r["alpha"], r["model_number"])))

    chunk_fields = [
        "model_number",
        "model",
        "alpha",
        "resolution",
        "chunk_index",
        "chunks_total",
        "start_index",
        "count",
        "status",
        "correct_y",
        "correct_x",
        "accy",
        "accx",
        "arena_bytes",
        "flash_seconds",
        "log",
    ]
    with CHUNKS_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=chunk_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(chunk_rows, key=lambda r: (r["resolution"], r["alpha"], r["model_number"], r["chunk_index"])))

    RUN_STATE_JSON.write_text(json.dumps({"models": model_rows, "chunks": chunk_rows}, indent=2), encoding="utf-8")


def summarize_model(base: dict, chunk_rows: list[dict], chunks_total: int, expected_count: int = N_IMAGES):
    done = [row for row in chunk_rows if row["model"] == base["model"] and row["status"] == "ok"]
    count = sum(int(row["count"]) for row in done)
    correct_y = sum(int(row["correct_y"]) for row in done)
    correct_x = sum(int(row["correct_x"]) for row in done)
    row = dict(base)
    row["chunks_done"] = len(done)
    row["chunks_total"] = chunks_total
    row["eval_count"] = count
    row["correct_y"] = correct_y
    row["correct_x"] = correct_x
    row["accy"] = correct_y / count if count else ""
    row["accx"] = correct_x / count if count else ""
    row["status"] = "ok" if count == expected_count and len(done) == chunks_total else "partial"
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=PORT)
    parser.add_argument("--flash-baud", type=int, default=FLASH_BAUD)
    parser.add_argument("--repl-baud", type=int, default=REPL_BAUD)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--model", action="append", help="Model number or filename substring.")
    parser.add_argument("--limit-images", type=int, default=N_IMAGES)
    args = parser.parse_args()
    if args.limit_images < 1 or args.limit_images > N_IMAGES:
        raise SystemExit("--limit-images must be between 1 and %d" % N_IMAGES)

    STAGE_DIR.mkdir(exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)
    models = discover_models(args.model)
    state = load_state(RUN_STATE_JSON) if args.resume else {"models": [], "chunks": []}
    model_rows = state["models"]
    chunk_rows = state["chunks"]

    print("Accuracy-only validation")
    print("Source models: %s" % SOURCE_DIR)
    print("Order: %s" % ", ".join(path.name for path in models))

    for model_path in models:
        number = model_number(model_path)
        alpha = alpha_for_model(number)
        size = resolution_for_model(number)
        root, txt, rows = read_validation_rows(size)
        rows = rows[:args.limit_images]
        meta = _tflite_meta.meta(model_path)
        stride = S.input_stride(meta)
        arena = arena_for(alpha, size)
        chunk_size = min(args.limit_images, initial_chunk_size(meta["file_bytes"], size))

        base = {
            "model_number": number,
            "model": model_path.name,
            "alpha": alpha,
            "resolution": size,
            "arena_bytes": arena,
            "validation_txt": str(txt),
            "validation_limit_images": args.limit_images,
        }

        print("\nMODEL %s alpha=%s res=%d arena=%d initial_chunk=%d" % (
            model_path.name, alpha, size, arena, chunk_size
        ), flush=True)

        staged_model, patches = stage_model(model_path)
        base["softmax_patch_count"] = len(patches)

        while True:
            chunks = make_chunks(rows, chunk_size)
            if not chunks:
                raise RuntimeError("No chunks")
            # Probe only the first largest chunk locally to ensure VFS fit.
            images_path, labels_path, _ = write_validation_chunk(size, chunks[0], 0)
            if try_build_image(staged_model, images_path, labels_path):
                break
            if chunk_size == 1:
                raise RuntimeError("Even one image does not fit with %s" % model_path.name)
            chunk_size = max(1, chunk_size // 2)
            print("  reducing chunk size to %d" % chunk_size, flush=True)

        chunks_total = len(chunks)
        model_rows = [row for row in model_rows if row.get("model") != model_path.name]
        model_rows.append(summarize_model(base, chunk_rows, chunks_total, args.limit_images))
        write_outputs(model_rows, chunk_rows)

        for chunk_index, chunk in enumerate(chunks):
            start_index = int(chunk[0][0])
            existing = next(
                (
                    row for row in chunk_rows
                    if row.get("model") == model_path.name
                    and int(row.get("chunk_index", -1)) == chunk_index
                    and row.get("status") == "ok"
                ),
                None,
            )
            if args.resume and existing:
                print("  skip chunk %d/%d already ok" % (chunk_index + 1, chunks_total), flush=True)
                continue

            print("  chunk %d/%d start=%d count=%d" % (
                chunk_index + 1, chunks_total, start_index, len(chunk)
            ), flush=True)
            images_path, labels_path, manifest = write_validation_chunk(size, chunk, chunk_index)
            while not try_build_image(staged_model, images_path, labels_path):
                if len(chunk) == 1:
                    raise RuntimeError("Single-image chunk failed to fit")
                # Re-split from this point onward with half the size.
                chunk_size = max(1, len(chunk) // 2)
                chunks = make_chunks(rows, chunk_size)
                chunks_total = len(chunks)
                print("  changed chunking to %d images/chunk; restart model on resume" % chunk_size)
                raise SystemExit(3)

            flash_seconds = flash_current_image(args.port, args.flash_baud)
            log_path = LOG_DIR / ("%s_chunk%03d.log" % (model_path.stem, chunk_index))
            time.sleep(0.8)
            data = run_chunk_eval(args.port, args.repl_baud, arena, stride, len(chunk), log_path)
            chunk_row = {
                **base,
                "chunk_index": chunk_index,
                "chunks_total": chunks_total,
                "start_index": start_index,
                "count": data.get("count", len(chunk)),
                "status": data.get("status"),
                "correct_y": data.get("correct_y", 0),
                "correct_x": data.get("correct_x", 0),
                "accy": data.get("accy", 0),
                "accx": data.get("accx", 0),
                "flash_seconds": round(flash_seconds, 3),
                "log": str(log_path),
                "manifest": str(manifest),
            }
            chunk_rows = [
                row for row in chunk_rows
                if not (
                    row.get("model") == model_path.name
                    and int(row.get("chunk_index", -1)) == chunk_index
                )
            ]
            chunk_rows.append(chunk_row)
            model_rows = [row for row in model_rows if row.get("model") != model_path.name]
            model_summary = summarize_model(base, chunk_rows, chunks_total, args.limit_images)
            model_rows.append(model_summary)
            write_outputs(model_rows, chunk_rows)
            print(
                "    chunk status=%s accy=%s accx=%s | model count=%s accy=%s accx=%s"
                % (
                    chunk_row["status"],
                    chunk_row["accy"],
                    chunk_row["accx"],
                    model_summary["eval_count"],
                    model_summary["accy"],
                    model_summary["accx"],
                ),
                flush=True,
            )
            if chunk_row["status"] != "ok":
                print("  stopping after failed chunk; rerun with --resume after fixing", flush=True)
                return 2

    write_outputs(model_rows, chunk_rows)
    print("\nWROTE %s" % SUMMARY_CSV)
    print("WROTE %s" % CHUNKS_CSV)
    print("WROTE %s" % RESULTS_JSON)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
