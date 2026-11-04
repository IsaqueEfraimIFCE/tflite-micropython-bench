import gc
import io
import os
import time

import microlite

try:
    import ujson as json
except ImportError:
    import json

try:
    import esp32
    _HAS_ESP32 = True
except ImportError:
    _HAS_ESP32 = False


MODEL_PATH = "model.tflite"
N_CLASSES = 5

_BUF = None
_INTERP = None
_INPUT = None


def _noop(interp):
    pass


def cleanup():
    global _BUF, _INTERP, _INPUT
    _INPUT = None
    _INTERP = None
    _BUF = None
    gc.collect()


def _heap():
    gc.collect()
    return gc.mem_free(), gc.mem_alloc()


def _idf_heap():
    if not _HAS_ESP32:
        return {"regions": 0, "free_bytes": 0, "largest_free_block": 0}
    regions = esp32.idf_heap_info(esp32.HEAP_DATA)
    return {
        "regions": len(regions),
        "free_bytes": sum(r[1] for r in regions),
        "largest_free_block": max((r[2] for r in regions), default=0),
    }


def _vfs():
    try:
        st = os.statvfs("/")
    except Exception:
        return {}
    bsize = int(st[0])
    frsize = int(st[1])
    blocks = int(st[2])
    bfree = int(st[3])
    bavail = int(st[4])
    unit = frsize or bsize
    return {
        "block_size": bsize,
        "fragment_size": frsize,
        "blocks": blocks,
        "free_blocks": bfree,
        "avail_blocks": bavail,
        "total_bytes": blocks * unit,
        "free_bytes": bfree * unit,
        "avail_bytes": bavail * unit,
        "used_bytes": (blocks - bfree) * unit,
    }


def _fill_input(image_stride, value):
    setv = _INPUT.setValue
    for i in range(image_stride):
        setv(i, value)


def _argmax(tensor):
    best_i = 0
    best_v = tensor.getValue(0)
    for i in range(1, N_CLASSES):
        v = tensor.getValue(i)
        if v > best_v:
            best_i = i
            best_v = v
    return best_i


def _init(arena_size, image_stride, warmup):
    global _BUF, _INTERP, _INPUT
    cleanup()

    f0_free, f0_alloc = _heap()
    idf0 = _idf_heap()
    vfs0 = _vfs()

    try:
        fsize = os.stat(MODEL_PATH)[6]
        _BUF = bytearray(fsize)
        with io.open(MODEL_PATH, "rb") as f:
            f.readinto(_BUF)
    except MemoryError:
        cleanup()
        return None, {
            "status": "mem_fail",
            "stage": "model_alloc",
            "arena_requested": arena_size,
            "heap_free_baseline": f0_free,
            "heap_alloc_baseline": f0_alloc,
            "vfs": vfs0,
        }

    f1_free, f1_alloc = _heap()

    try:
        _INTERP = microlite.interpreter(_BUF, arena_size, _noop, _noop)
    except MemoryError:
        cleanup()
        return None, {
            "status": "mem_fail",
            "stage": "interpreter",
            "file_bytes": fsize,
            "arena_requested": arena_size,
            "ram_model_bytes": f0_free - f1_free,
            "heap_free_baseline": f0_free,
            "heap_free_postmodel": f1_free,
            "heap_alloc_baseline": f0_alloc,
            "heap_alloc_postmodel": f1_alloc,
            "vfs": vfs0,
        }
    except Exception as exc:
        cleanup()
        return None, {
            "status": "init_fail",
            "stage": "interpreter",
            "error": repr(exc),
            "file_bytes": fsize,
            "arena_requested": arena_size,
            "ram_model_bytes": f0_free - f1_free,
            "heap_free_baseline": f0_free,
            "heap_free_postmodel": f1_free,
            "heap_alloc_baseline": f0_alloc,
            "heap_alloc_postmodel": f1_alloc,
            "vfs": vfs0,
        }

    _INPUT = _INTERP.getInputTensor(0)
    f2_free, f2_alloc = _heap()
    idf2 = _idf_heap()

    stats = {
        "status": "ok",
        "file_bytes": fsize,
        "arena_requested": arena_size,
        "ram_model_bytes": f0_free - f1_free,
        "ram_arena_bytes": f1_free - f2_free,
        "ram_total_bytes": f0_free - f2_free,
        "heap_free_baseline": f0_free,
        "heap_free_postmodel": f1_free,
        "heap_free_postinterp": f2_free,
        "heap_alloc_baseline": f0_alloc,
        "heap_alloc_postmodel": f1_alloc,
        "heap_alloc_postinterp": f2_alloc,
        "idf_dram_free_baseline": idf0["free_bytes"],
        "idf_dram_lfb_baseline": idf0["largest_free_block"],
        "idf_dram_free_postinterp": idf2["free_bytes"],
        "idf_dram_lfb_postinterp": idf2["largest_free_block"],
        "vfs": vfs0,
        "model_storage": "vfs_file_loaded_to_ram_bytearray",
        "model_in_ram": True,
    }

    if warmup:
        try:
            _fill_input(image_stride, 0)
            _INTERP.invoke()
        except Exception as exc:
            stats["status"] = "invoke_fail"
            stats["error"] = repr(exc)
            cleanup()
            return None, stats
        f3_free, f3_alloc = _heap()
        stats["heap_free_postinvoke"] = f3_free
        stats["heap_alloc_postinvoke"] = f3_alloc
        stats["invoke_alloc_delta"] = f3_alloc - f2_alloc

    return _INTERP, stats


def probe(arena_size, image_stride):
    interp, stats = _init(arena_size, image_stride, True)
    print("PROBE_JSON:" + json.dumps(stats))
    cleanup()


def profile(arena_size, image_stride, n_iters):
    interp, stats = _init(arena_size, image_stride, True)
    if interp is None:
        print("PROFILE_JSON:" + json.dumps(stats))
        return

    total_us = 0
    min_us = 1 << 30
    max_us = 0
    for _ in range(n_iters):
        t0 = time.ticks_us()
        interp.invoke()
        dt = time.ticks_diff(time.ticks_us(), t0)
        total_us += dt
        if dt < min_us:
            min_us = dt
        if dt > max_us:
            max_us = dt

    stats["avg_invoke_us"] = total_us // n_iters
    stats["min_invoke_us"] = min_us
    stats["max_invoke_us"] = max_us
    stats["iters"] = n_iters
    print("PROFILE_JSON:" + json.dumps(stats))
    cleanup()


def evaluate_float(arena_size, images_path, labels_path, image_stride, n_images):
    """Evaluate FLOAT32-input MobileNet on uint8 images using mobilenet_pm1
    preprocessing: f = (uint8 - 127.5) / 127.5  (range -1..1).
    Output mapping (matches host probe): output 0 -> x, output 1 -> y."""
    interp, stats = _init(arena_size, image_stride, False)
    if interp is None:
        print("EVAL_JSON:" + json.dumps(stats))
        return

    label_y = bytearray(n_images)
    label_x = bytearray(n_images)
    with io.open(labels_path, "rb") as f:
        f.readinto(label_y)
        f.readinto(label_x)

    out0 = interp.getOutputTensor(0)
    out1 = interp.getOutputTensor(1)
    setv = _INPUT.setValue
    img = bytearray(image_stride)

    correct_y = 0
    correct_x = 0
    total_us = 0
    min_us = 1 << 30
    max_us = 0

    inv_127_5 = 1.0 / 127.5

    with io.open(images_path, "rb") as f:
        for idx in range(n_images):
            f.readinto(img)
            for i in range(image_stride):
                setv(i, (img[i] - 127.5) * inv_127_5)

            t0 = time.ticks_us()
            interp.invoke()
            dt = time.ticks_diff(time.ticks_us(), t0)
            total_us += dt
            if dt < min_us:
                min_us = dt
            if dt > max_us:
                max_us = dt

            pred_x = _argmax(out0)
            pred_y = _argmax(out1)
            if pred_x == label_x[idx]:
                correct_x += 1
            if pred_y == label_y[idx]:
                correct_y += 1

            if (idx + 1) % 25 == 0:
                print("EVAL_PROGRESS %d/%d" % (idx + 1, n_images))

    stats["eval_count"] = n_images
    stats["correct_y"] = correct_y
    stats["correct_x"] = correct_x
    stats["accy"] = correct_y / n_images
    stats["accx"] = correct_x / n_images
    stats["avg_invoke_us"] = total_us // n_images
    stats["min_invoke_us"] = min_us
    stats["max_invoke_us"] = max_us
    stats["iters"] = n_images
    stats["preproc"] = "mobilenet_pm1"
    print("EVAL_JSON:" + json.dumps(stats))
    cleanup()


def evaluate(arena_size, images_path, labels_path, image_stride, n_images):
    interp, stats = _init(arena_size, image_stride, False)
    if interp is None:
        print("EVAL_JSON:" + json.dumps(stats))
        return

    label_y = bytearray(n_images)
    label_x = bytearray(n_images)
    with io.open(labels_path, "rb") as f:
        f.readinto(label_y)
        f.readinto(label_x)

    out0 = interp.getOutputTensor(0)
    out1 = interp.getOutputTensor(1)
    setv = _INPUT.setValue
    img = bytearray(image_stride)

    correct_y = 0
    correct_x = 0
    total_us = 0
    min_us = 1 << 30
    max_us = 0

    with io.open(images_path, "rb") as f:
        for idx in range(n_images):
            f.readinto(img)
            for i in range(image_stride):
                setv(i, img[i])

            t0 = time.ticks_us()
            interp.invoke()
            dt = time.ticks_diff(time.ticks_us(), t0)
            total_us += dt
            if dt < min_us:
                min_us = dt
            if dt > max_us:
                max_us = dt

            # Established mapping for these models: output0 -> x, output1 -> y.
            pred_x = _argmax(out0)
            pred_y = _argmax(out1)
            if pred_x == label_x[idx]:
                correct_x += 1
            if pred_y == label_y[idx]:
                correct_y += 1

            if (idx + 1) % 25 == 0:
                print("EVAL_PROGRESS %d/%d" % (idx + 1, n_images))

    stats["eval_count"] = n_images
    stats["correct_y"] = correct_y
    stats["correct_x"] = correct_x
    stats["accy"] = correct_y / n_images
    stats["accx"] = correct_x / n_images
    stats["avg_invoke_us"] = total_us // n_images
    stats["min_invoke_us"] = min_us
    stats["max_invoke_us"] = max_us
    stats["iters"] = n_images
    print("EVAL_JSON:" + json.dumps(stats))
    cleanup()
