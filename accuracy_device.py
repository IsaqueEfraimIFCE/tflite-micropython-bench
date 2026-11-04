import gc
import io

import microlite

try:
    import ujson as json
except ImportError:
    import json


MODEL_PATH = "model.tflite"
N_CLASSES = 5


def _noop(interp):
    pass


def _argmax(tensor):
    best_i = 0
    best_v = tensor.getValue(0)
    for i in range(1, N_CLASSES):
        value = tensor.getValue(i)
        if value > best_v:
            best_i = i
            best_v = value
    return best_i


def evaluate(arena_size, image_stride, n_images, images_path="images.bin", labels_path="labels.bin"):
    buf = None
    interp = None
    try:
        model_size = __import__("os").stat(MODEL_PATH)[6]
        buf = bytearray(model_size)
        with io.open(MODEL_PATH, "rb") as f:
            f.readinto(buf)
        interp = microlite.interpreter(buf, arena_size, _noop, _noop)
        input_tensor = interp.getInputTensor(0)
        out0 = interp.getOutputTensor(0)
        out1 = interp.getOutputTensor(1)

        label_y = bytearray(n_images)
        label_x = bytearray(n_images)
        with io.open(labels_path, "rb") as f:
            f.readinto(label_y)
            f.readinto(label_x)

        image = bytearray(image_stride)
        setv = input_tensor.setValue
        correct_y = 0
        correct_x = 0

        with io.open(images_path, "rb") as f:
            for idx in range(n_images):
                f.readinto(image)
                for i in range(image_stride):
                    setv(i, image[i])
                interp.invoke()

                # Converted uint8 models use output0 -> accx, output1 -> accy.
                pred_x = _argmax(out0)
                pred_y = _argmax(out1)
                if pred_y == label_y[idx]:
                    correct_y += 1
                if pred_x == label_x[idx]:
                    correct_x += 1
                if (idx + 1) % 25 == 0:
                    print("ACC_PROGRESS %d/%d" % (idx + 1, n_images))

        print("ACC_JSON:" + json.dumps({
            "status": "ok",
            "count": n_images,
            "correct_y": correct_y,
            "correct_x": correct_x,
            "accy": correct_y / n_images,
            "accx": correct_x / n_images,
        }))
    except Exception as exc:
        print("ACC_JSON:" + json.dumps({
            "status": "error",
            "error": repr(exc),
            "count": n_images,
            "correct_y": 0,
            "correct_x": 0,
            "accy": 0,
            "accx": 0,
        }))
    finally:
        interp = None
        buf = None
        gc.collect()
