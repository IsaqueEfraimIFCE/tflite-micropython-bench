"""Shared helpers for ESP32-S3 MicroPython VFS profiling runners."""

import json
import re
import time
from pathlib import Path

import run_patched_tflite_bench as R


MODELS_DIR = Path(__file__).resolve().parent
FIRMWARE_BUILD = MODELS_DIR / "firmware_artifacts" / "s3_spiram_octal"
ARENA_RESOLUTION = 500
PROFILE_ITERS = 3
IMAGE_C = 3


def firmware_sizes():
    boot = FIRMWARE_BUILD / "bootloader" / "bootloader.bin"
    part = FIRMWARE_BUILD / "partition_table" / "partition-table.bin"
    app = FIRMWARE_BUILD / "micropython.bin"
    sizes = {
        "bootloader_bin_bytes": boot.stat().st_size if boot.exists() else None,
        "partition_table_bin_bytes": part.stat().st_size if part.exists() else None,
        "micropython_bin_bytes": app.stat().st_size if app.exists() else None,
    }
    present = [value for value in sizes.values() if value is not None]
    sizes["firmware_flash_artifact_bytes"] = sum(present) if len(present) == 3 else None
    return sizes


def parse_model_name(name):
    match = re.search(r"_a([0-9.]+)_rep(\d+)_(\d+)_", name)
    if not match:
        return {"alpha": "", "rep": "", "input_size": ""}
    return {
        "alpha": match.group(1),
        "rep": int(match.group(2)),
        "input_size": int(match.group(3)),
    }


def input_stride(meta):
    shape = meta["inputs"][0]["shape"]
    stride = 1
    for dim in shape[1:]:
        stride *= dim
    return stride


def repl_call(ser, code, marker, timeout, log_file=None):
    R._seq[0] += 1
    sentinel = b"__SY%d__" % R._seq[0]
    ser.write(code.encode("utf-8") + b';print("' + sentinel + b'")\r\n')
    out = b""
    end = time.time() + timeout
    marker_bytes = marker.encode("utf-8")
    while time.time() < end:
        chunk = ser.read(4096)
        if chunk:
            out += chunk
            if log_file is not None:
                log_file.write(chunk.decode("utf-8", "replace"))
                log_file.flush()
            if marker_bytes in out and out.count(sentinel) >= 2:
                break
            if b"Traceback" in out and out.count(sentinel) >= 2:
                break
        else:
            time.sleep(0.005)
    return out.decode("utf-8", "replace")


def json_from_output(text, prefix):
    found = None
    for line in text.splitlines():
        if prefix in line:
            payload = line.split(prefix, 1)[1].strip()
            try:
                found = json.loads(payload)
            except ValueError:
                pass
    return found


def probe_timeout_for(stride, file_bytes):
    pixels = stride // IMAGE_C
    if pixels <= 64 * 64:
        return 35
    if pixels <= 128 * 128:
        return 60 if file_bytes <= 1_200_000 else 120
    if file_bytes <= 400_000:
        return 45
    if file_bytes <= 1_200_000:
        return 75
    return 180


def probe_arena(ser, arena, stride, file_bytes, timeout=None):
    if timeout is None:
        timeout = probe_timeout_for(stride, file_bytes)
    R.reset_and_repl(ser)
    code = "import profile_device; profile_device.probe(%d, %d)" % (arena, stride)
    text = repl_call(ser, code, "PROBE_JSON:", timeout)
    data = json_from_output(text, "PROBE_JSON:")
    if data is None:
        return {"status": "timeout", "arena_requested": arena, "tail": text[-600:]}
    return data


def candidate_arenas(file_bytes):
    max_reasonable = max(750_000, min(4_500_000, 7_700_000 - file_bytes))
    if file_bytes <= 400_000:
        values = [600_000, 900_000, 1_200_000]
    elif file_bytes <= 1_200_000:
        values = [1_500_000, 2_200_000, 3_000_000]
    else:
        values = [3_200_000, 3_800_000, 4_200_000]
    return [value for value in values if value <= max_reasonable]


def find_min_arena(ser, stride, file_bytes):
    history = []
    hi = None
    low = 0
    for arena in candidate_arenas(file_bytes):
        print("  find-hi probe %d" % arena, flush=True)
        result = probe_arena(ser, arena, stride, file_bytes)
        status = result.get("status", "unknown")
        history.append({"arena": arena, "status": status, "stage": result.get("stage")})
        print("    %s" % status, flush=True)
        if status == "ok":
            hi = arena
            break
        low = arena

    if hi is None:
        return None, history

    while hi - low > ARENA_RESOLUTION:
        mid = (low + hi) // 2
        print("  binary probe %d" % mid, flush=True)
        result = probe_arena(ser, mid, stride, file_bytes)
        status = result.get("status", "unknown")
        history.append({"arena": mid, "status": status, "stage": result.get("stage")})
        print("    %s" % status, flush=True)
        if status == "ok":
            hi = mid
        else:
            low = mid

    final = probe_arena(ser, hi, stride, file_bytes)
    history.append({"arena": hi, "status": final.get("status", "unknown"), "stage": "final_verify"})
    if final.get("status") != "ok":
        return None, history
    return hi, history


def run_profile(ser, arena, stride, n_iters=PROFILE_ITERS):
    R.reset_and_repl(ser)
    code = "import profile_device; profile_device.profile(%d, %d, %d)" % (arena, stride, n_iters)
    text = repl_call(ser, code, "PROFILE_JSON:", timeout=1200)
    data = json_from_output(text, "PROFILE_JSON:")
    if data is None:
        return {"status": "timeout", "tail": text[-600:]}
    return data


def flatten_runtime(prefix, data, row):
    row[prefix + "status"] = data.get("status")
    for key in (
        "arena_requested",
        "file_bytes",
        "ram_model_bytes",
        "ram_arena_bytes",
        "ram_total_bytes",
        "heap_free_baseline",
        "heap_free_postmodel",
        "heap_free_postinterp",
        "heap_free_postinvoke",
        "heap_alloc_baseline",
        "heap_alloc_postmodel",
        "heap_alloc_postinterp",
        "heap_alloc_postinvoke",
        "invoke_alloc_delta",
        "idf_dram_free_baseline",
        "idf_dram_lfb_baseline",
        "idf_dram_free_postinterp",
        "idf_dram_lfb_postinterp",
        "avg_invoke_us",
        "min_invoke_us",
        "max_invoke_us",
        "iters",
    ):
        row[prefix + key] = data.get(key)

    vfs = data.get("vfs") or {}
    row[prefix + "vfs_total_bytes"] = vfs.get("total_bytes")
    row[prefix + "vfs_used_bytes"] = vfs.get("used_bytes")
    row[prefix + "vfs_free_bytes"] = vfs.get("free_bytes")
    row[prefix + "model_in_ram"] = data.get("model_in_ram")
    row[prefix + "model_storage"] = data.get("model_storage")


def add_flash_totals(row, runtime, fw):
    vfs = (runtime.get("vfs") or {}).get("used_bytes")
    artifacts = fw.get("firmware_flash_artifact_bytes")
    row["full_flash_used_bytes"] = artifacts + vfs if artifacts is not None and vfs is not None else None
    row.update(fw)


def model_row_base(model_path, meta, patches):
    parsed = parse_model_name(model_path.name)
    in0 = meta["inputs"][0]
    out_shapes = ["x".join(str(x) for x in out["shape"]) for out in meta["outputs"]]
    return {
        "name": model_path.name,
        "alpha": parsed["alpha"],
        "rep": parsed["rep"],
        "input_size": parsed["input_size"],
        "input_shape": "x".join(str(x) for x in in0["shape"]),
        "input_dtype": in0["dtype"],
        "output_shapes": ";".join(out_shapes),
        "output_dtypes": ";".join(out["dtype"] for out in meta["outputs"]),
        "model_file_bytes": meta["file_bytes"],
        "param_bytes": meta["param_bytes"],
        "op_count": meta["op_count"],
        "tensor_count": meta["tensor_count"],
        "softmax_patch_count": len(patches),
        "softmax_patches": json.dumps(patches, separators=(",", ":")),
    }
