"""Current/power-probe harness for a single TFLite Micro inference.

Goal: measure the current the ESP32-S3 draws *only while* interp.invoke()
runs, with every non-essential subsystem quieted so it does not pollute the
trace. Pair with an external current analyzer (PPK2 / Otii / Joulescope / INA226
shunt) on the 3V3 rail, triggered by the marker GPIO.

Device usage (over the REPL, same pattern as profile_device):
    import power_probe
    power_probe.run(85742, 3072)          # arena_size, image_stride
    power_probe.run(85742, 3072, iters=200, marker_pin=21, idle_ms=2000)

Wiring:
    Connect MARKER_PIN to your analyzer's digital/trigger input. It goes HIGH
    for the whole invoke() window and LOW between iterations, so each rising
    edge brackets exactly one inference. An optional idle phase at the start
    (LED/marker LOW, CPU spinning in a no-op loop) gives you a baseline to
    subtract regulator + bridge quiescent current.
"""
import gc
import io
import os
import time

import microlite

try:
    import ujson as json
except ImportError:
    import json

from machine import Pin, freq, deepsleep

try:
    import network
    _HAS_NET = True
except ImportError:
    _HAS_NET = False

try:
    import bluetooth
    _HAS_BT = True
except ImportError:
    _HAS_BT = False


MODEL_PATH = "model.tflite"

# Most ESP32-S3 devkits wire the onboard RGB (neopixel) LED to GPIO48.
LED_PIN = 48
DEFAULT_MARKER_PIN = 21
CPU_HZ = 240_000_000


def _noop(interp):
    pass


def _quiet_peripherals(cpu_hz=CPU_HZ):
    """Disable radios + onboard LED and pin CPU frequency. Returns notes."""
    notes = {}

    if _HAS_NET:
        try:
            network.WLAN(network.STA_IF).active(False)
            network.WLAN(network.AP_IF).active(False)
            notes["wifi"] = "off"
        except Exception as exc:
            notes["wifi"] = repr(exc)

    if _HAS_BT:
        try:
            bluetooth.BLE().active(False)
            notes["ble"] = "off"
        except Exception as exc:
            notes["ble"] = repr(exc)

    try:
        Pin(LED_PIN, Pin.OUT).value(0)
        notes["led"] = "low"
    except Exception as exc:
        notes["led"] = repr(exc)

    try:
        freq(cpu_hz)
        notes["cpu_hz"] = freq()
    except Exception as exc:
        notes["cpu_hz"] = repr(exc)

    return notes


def run(arena_size, image_stride, iters=200, marker_pin=DEFAULT_MARKER_PIN,
        idle_ms=0, fill=0, cpu_hz=CPU_HZ):
    """Load the model, quiet the board, then pulse the marker GPIO around each
    of `iters` invoke() calls. No printing happens inside the timed window."""
    marker = Pin(marker_pin, Pin.OUT)
    marker.value(0)

    notes = _quiet_peripherals(cpu_hz)

    buf = None
    interp = None
    try:
        fsize = os.stat(MODEL_PATH)[6]
        buf = bytearray(fsize)
        with io.open(MODEL_PATH, "rb") as f:
            f.readinto(buf)
        interp = microlite.interpreter(buf, arena_size, _noop, _noop)
        inp = interp.getInputTensor(0)

        setv = inp.setValue
        for i in range(image_stride):
            setv(i, fill)

        # Warm-up invoke (lazy allocs, caches) outside the measured window.
        interp.invoke()

        # Optional idle baseline: marker LOW, CPU just spins so the analyzer
        # captures quiescent current to subtract later.
        if idle_ms:
            t_end = time.ticks_add(time.ticks_ms(), idle_ms)
            while time.ticks_diff(t_end, time.ticks_ms()) > 0:
                pass

        gc.collect()
        gc.disable()

        total_us = 0
        min_us = 1 << 30
        max_us = 0
        invoke = interp.invoke
        for _ in range(iters):
            marker.value(1)
            t0 = time.ticks_us()
            invoke()
            dt = time.ticks_diff(time.ticks_us(), t0)
            marker.value(0)
            total_us += dt
            if dt < min_us:
                min_us = dt
            if dt > max_us:
                max_us = dt

        gc.enable()

        print("POWER_JSON:" + json.dumps({
            "status": "ok",
            "arena": arena_size,
            "image_stride": image_stride,
            "iters": iters,
            "marker_pin": marker_pin,
            "idle_ms": idle_ms,
            "avg_invoke_us": total_us // iters,
            "min_invoke_us": min_us,
            "max_invoke_us": max_us,
            "file_bytes": fsize,
            "quiet": notes,
        }))
    except Exception as exc:
        try:
            gc.enable()
        except Exception:
            pass
        print("POWER_JSON:" + json.dumps({"status": "error", "error": repr(exc),
                                          "quiet": notes}))
    finally:
        marker.value(0)
        interp = None
        buf = None
        gc.collect()


def deep_sleep_baseline(sleep_ms=10000, marker_pin=DEFAULT_MARKER_PIN,
                        cpu_hz=CPU_HZ):
    """Absolute floor: quiet every subsystem, drop the marker LOW, then put the
    ESP32-S3 into deep sleep (both cores + most RAM + radios powered down) for
    `sleep_ms`. The CPU is off during the window, so nothing is printed and no
    GPIO toggles -- the current trace itself is the marker (a hard drop to the
    RTC-only floor). Timer wake after sleep_ms triggers a full reset/reboot.

    The host must detect the pre-sleep BASELINE_JSON, hold the analyzer window
    for sleep_ms, then catch the reboot banner."""
    try:
        Pin(marker_pin, Pin.OUT).value(0)
    except Exception:
        pass
    notes = _quiet_peripherals(cpu_hz)
    print("BASELINE_JSON:" + json.dumps({
        "status": "sleeping",
        "mode": "deep_sleep",
        "sleep_ms": sleep_ms,
        "marker_pin": marker_pin,
        "quiet": notes,
    }))
    # Let the UART FIFO drain before the cores power down.
    time.sleep_ms(60)
    deepsleep(sleep_ms)  # both cores off; returns via reset on timer wake
