import argparse
import base64
import json
import re
import time
from pathlib import Path

import serial
import tflite


PORT = "COM12"
BAUD = 115200
BAUD_FAST = 460800        # bumped REPL baud for fast upload (CH343 reliable up to 1.5M; raw-paste protocol caps ~11 KB/s past 460800 anyway)
CHUNK = 1024              # legacy friendly-REPL chunk size
CHUNK_FAST = 8192         # binary bytes per f.write() inside raw-paste script
GC_EVERY = 64
SOFTMAX_MIN_SCALE = 2e-8

_seq = [0]
_baud_state = ["lo"]


def drain(ser, seconds):
    end = time.time() + seconds
    out = b""
    while time.time() < end:
        data = ser.read(4096)
        if data:
            out += data
        else:
            time.sleep(0.01)
    return out


def reset_and_repl(ser):
    ser.setDTR(False)
    ser.setRTS(True)
    time.sleep(0.12)
    ser.setRTS(False)
    time.sleep(0.5)
    drain(ser, 1.5)
    ser.write(b"\r\x03\x03\x02\r")
    time.sleep(0.25)
    drain(ser, 0.6)
    ser.write(b"\r")
    drain(ser, 0.4)


def cmd(ser, line, timeout=3.0):
    _seq[0] += 1
    sentinel = b"__SY%d__" % _seq[0]
    ser.write(line + b';print("' + sentinel + b'")\r\n')
    out = b""
    end = time.time() + timeout
    while time.time() < end:
        data = ser.read(4096)
        if data:
            out += data
            if out.count(sentinel) >= 2:
                return out
        else:
            time.sleep(0.002)
    raise TimeoutError("no sync after %r" % line[:80])


def patch_softmax_scales(data):
    model = tflite.Model.GetRootAsModel(data, 0)
    graph = model.Subgraphs(0)
    builtin = {
        getattr(tflite.BuiltinOperator, name): name
        for name in dir(tflite.BuiltinOperator)
        if name.isupper() and isinstance(getattr(tflite.BuiltinOperator, name), int)
    }
    patched = []
    for op_index in range(graph.OperatorsLength()):
        op = graph.Operators(op_index)
        opcode = model.OperatorCodes(op.OpcodeIndex())
        if builtin.get(opcode.BuiltinCode()) != "SOFTMAX":
            continue
        tensor_index = op.Inputs(0)
        quant = graph.Tensors(tensor_index).Quantization()
        if not quant or quant.ScaleLength() == 0:
            continue
        scales = quant.ScaleAsNumpy()
        old_scale = float(scales[0])
        if old_scale * (1 << 26) <= 1.0:
            scales[0] = SOFTMAX_MIN_SCALE
            patched.append(
                {
                    "op": op_index,
                    "tensor": tensor_index,
                    "old_scale": old_scale,
                    "new_scale": float(scales[0]),
                }
            )
    return patched


def _enter_raw(ser):
    ser.write(b"\r\x03\x03\r\x01")
    end = time.time() + 2
    out = b""
    while time.time() < end:
        out += ser.read(4096)
        if b"raw REPL; CTRL-B to exit\r\n>" in out:
            return
    raise RuntimeError("no raw REPL prompt: %r" % out[-200:])


def _exit_raw(ser):
    ser.write(b"\r\x02")
    drain(ser, 0.3)


def _raw_exec_ack(ser, code):
    """Send code + Ctrl-D in raw REPL; return after device sends 'OK'."""
    ser.write(code + b"\x04")
    out = b""
    end = time.time() + 5
    while time.time() < end:
        out += ser.read(4096)
        if b"OK" in out:
            return out
    raise RuntimeError("no OK after raw exec: %r" % out[-200:])


def bump_baud(ser, hi=BAUD_FAST):
    """Switch device REPL UART + host pyserial to `hi`. Idempotent."""
    if _baud_state[0] == "hi" and ser.baudrate == hi:
        return
    _enter_raw(ser)
    _raw_exec_ack(ser, b"from machine import UART\nUART(0, %d)" % hi)
    time.sleep(0.1)
    ser.flush()
    ser.baudrate = hi
    time.sleep(0.1)
    drain(ser, 0.2)
    _enter_raw(ser)
    _exit_raw(ser)
    _baud_state[0] = "hi"


def restore_baud(ser, lo=BAUD):
    if _baud_state[0] == "lo" and ser.baudrate == lo:
        return
    try:
        _enter_raw(ser)
        _raw_exec_ack(ser, b"from machine import UART\nUART(0, %d)" % lo)
        time.sleep(0.1)
    except Exception:
        pass
    ser.flush()
    ser.baudrate = lo
    time.sleep(0.1)
    drain(ser, 0.2)
    _baud_state[0] = "lo"


def _raw_paste_exec(ser, script, output_timeout=300):
    """Execute `script` via MicroPython raw-paste mode. Returns response bytes."""
    ser.write(b"\x05A\x01")
    hdr = b""
    end = time.time() + 2
    while len(hdr) < 2 and time.time() < end:
        hdr += ser.read(2 - len(hdr))
    if hdr != b"R\x01":
        # Fallback: plain raw exec.
        ser.write(script + b"\x04")
        out = b""
        end = time.time() + output_timeout
        while time.time() < end:
            d = ser.read(4096)
            if d:
                out += d
                if out.count(b"\x04") >= 2:
                    return out
            else:
                time.sleep(0.005)
        raise RuntimeError("raw exec timeout (no paste support): %r" % out[-200:])
    win = b""
    end = time.time() + 1
    while len(win) < 2 and time.time() < end:
        win += ser.read(2 - len(win))
    window = int.from_bytes(win, "little")
    remain = window
    pos, n = 0, len(script)
    while pos < n:
        if remain == 0:
            ack = b""
            end = time.time() + 5
            while not ack and time.time() < end:
                ack = ser.read(1)
            if not ack:
                raise RuntimeError("paste ack timeout at %d" % pos)
            if ack == b"\x04":
                raise RuntimeError("paste aborted at %d" % pos)
            if ack == b"\x01":
                remain += window
            continue
        send = min(remain, n - pos, 1024)
        ser.write(script[pos:pos + send])
        pos += send
        remain -= send
        d = ser.read(ser.in_waiting or 0)
        for b in d:
            if b == 0x01:
                remain += window
            elif b == 0x04:
                raise RuntimeError("paste aborted mid at %d" % pos)
    ser.write(b"\x04")
    out = b""
    end = time.time() + output_timeout
    while time.time() < end:
        d = ser.read(4096)
        if d:
            out += d
            if out.count(b"\x04") >= 2:
                return out
        else:
            time.sleep(0.005)
    raise RuntimeError("paste output timeout: %r" % out[-200:])


def upload_bytes_fast(ser, data, remote, chunk=CHUNK_FAST):
    """Upload via raw-paste mode; assumes baud already bumped."""
    n = len(data)
    parts = [b"import ubinascii, os\n",
             ("try:\n os.remove('%s')\nexcept OSError:\n pass\n" % remote).encode(),
             ("f=open('%s','wb')\n" % remote).encode()]
    for i in range(0, n, chunk):
        b64 = base64.b64encode(data[i:i + chunk])
        parts.append(b"f.write(ubinascii.a2b_base64(b'" + b64 + b"'))\n")
    parts.append(b"f.close()\n")
    parts.append(("print('SZ', os.stat('%s')[6])\n" % remote).encode())
    script = b"".join(parts)
    _enter_raw(ser)
    t0 = time.time()
    out = _raw_paste_exec(ser, script, output_timeout=600)
    dt = time.time() - t0
    rate = n / max(0.001, dt) / 1024
    print("UPLOAD_FAST %d B in %.2fs = %.2f KB/s" % (n, dt, rate))
    _exit_raw(ser)
    if not re.search(rb"SZ\s+%d\b" % n, out):
        raise RuntimeError("fast upload size check failed: %r" % out[-200:])


def upload_bytes(ser, data, remote, chunk=CHUNK):
    cmd(ser, b"import ubinascii,os,gc")
    cmd(
        ser,
        ('exec("try:\\n os.remove(\'%s\')\\nexcept OSError:\\n pass")' % remote).encode(),
    )
    cmd(ser, ("f=open('%s','wb')" % remote).encode())
    start = time.time()
    last_print = start
    chunks = 0
    size = len(data)
    for offset in range(0, size, chunk):
        b64 = base64.b64encode(data[offset : offset + chunk])
        cmd(ser, b"f.write(ubinascii.a2b_base64(b'" + b64 + b"'))", timeout=12.0)
        chunks += 1
        if chunks % GC_EVERY == 0:
            cmd(ser, b"gc.collect()", timeout=3.0)
        if time.time() - last_print > 15:
            done = min(size, offset + chunk)
            rate = done / max(0.001, time.time() - start) / 1024
            print("UPLOAD %d/%d %.1f%% %.1fKB/s" % (done, size, 100 * done / size, rate))
            last_print = time.time()
    cmd(ser, b"f.close()")
    cmd(ser, b"gc.collect()", timeout=3.0)
    out = cmd(
        ser,
        ("import os; print('SZ', os.stat('%s')[6])" % remote).encode(),
        timeout=3.0,
    )
    if not re.search(rb"SZ\s+%d\b" % size, out):
        raise RuntimeError("upload size check failed: %r" % out[-200:])


def run_bench(ser, arena):
    cmd(ser, b"import sys; sys.modules.pop('bench', None)")
    _seq[0] += 1
    sentinel = b"__SY%d__" % _seq[0]
    ser.write(
        ("import bench; bench.run('model.tflite', %d)" % arena).encode()
        + b';print("'
        + sentinel
        + b'")\r\n'
    )
    out = b""
    end = time.time() + 180
    while time.time() < end:
        data = ser.read(4096)
        if data:
            out += data
            if b"HALTED" in out or out.count(sentinel) >= 2:
                break
        else:
            time.sleep(0.005)
    return out.decode("utf-8", "replace")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--arena", type=int, required=True)
    parser.add_argument("--upload-bench", type=Path)
    parser.add_argument("--port", default=PORT)
    parser.add_argument("--baud", type=int, default=BAUD)
    parser.add_argument("--chunk", type=int, default=CHUNK)
    parser.add_argument("--fast", action="store_true", default=True,
                        help="raw-paste + bumped baud upload (~11 KB/s, 2.6× baseline). On by default.")
    parser.add_argument("--slow", dest="fast", action="store_false",
                        help="legacy friendly-REPL upload (~4.3 KB/s).")
    parser.add_argument("--fast-baud", type=int, default=BAUD_FAST)
    parser.add_argument("--fast-chunk", type=int, default=CHUNK_FAST)
    args = parser.parse_args()

    model_bytes = bytearray(args.model.read_bytes())
    patched = patch_softmax_scales(model_bytes)
    print("MODEL", args.model.name, "bytes", len(model_bytes), "arena", args.arena)
    print("PATCHED", json.dumps(patched, separators=(",", ":")))
    print("UPLOAD_MODE", "fast" if args.fast else "slow")

    with serial.Serial(args.port, args.baud, timeout=0.05) as ser:
        reset_and_repl(ser)
        if args.fast:
            bump_baud(ser, args.fast_baud)
            if args.upload_bench:
                print("UPLOAD_BENCH", args.upload_bench)
                upload_bytes_fast(ser, args.upload_bench.read_bytes(), "bench.py", args.fast_chunk)
            upload_bytes_fast(ser, bytes(model_bytes), "model.tflite", args.fast_chunk)
        else:
            if args.upload_bench:
                print("UPLOAD_BENCH", args.upload_bench)
                upload_bytes(ser, args.upload_bench.read_bytes(), "bench.py", args.chunk)
            upload_bytes(ser, bytes(model_bytes), "model.tflite", args.chunk)
        print("RESET_AFTER_UPLOAD")
        reset_and_repl(ser)
        # reset returns the device to its build-time default baud (115200).
        ser.baudrate = args.baud
        _baud_state[0] = "lo"
        time.sleep(0.1)
        drain(ser, 0.3)
        out = run_bench(ser, args.arena)

    print("RAW_BEGIN")
    print(out)
    print("RAW_END")
    match = re.search(r"BENCH_JSON:(\{.*\})", out)
    if match:
        result = json.loads(match.group(1))
        result["status"] = "ok"
        result["name"] = args.model.name
        result["arena"] = args.arena
        print("RESULT_JSON:" + json.dumps(result, separators=(",", ":")))
        return 0
    status = "tflm_halted" if "HALTED" in out else "failed"
    print("RESULT_JSON:" + json.dumps({"name": args.model.name, "status": status}))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
