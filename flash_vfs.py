"""Build a LittleFS image with selected files and flash it to the VFS partition.

Replaces the slow REPL upload pipeline (~11 KB/s) with a direct esptool
flash (~300-400 KB/s). Wipes the current VFS partition contents.

Usage:
  python flash_vfs.py path/to/model.tflite [more_files...]
    [--port COM12] [--baud 460800] [--remote-name model.tflite]
    [--no-backup] [--dry-run]
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from littlefs import LittleFS


VFS_OFFSET = 0x200000
VFS_SIZE = 0x600000
BLOCK_SIZE = 4096
BLOCK_COUNT = VFS_SIZE // BLOCK_SIZE  # 1536
READ_SIZE = 32
PROG_SIZE = 32
LOOKAHEAD = 32
BLOCK_CYCLES = 100
DISK_VERSION = 0x00020000  # LittleFS 2.0 — matches MicroPython v1.19.1 submodule.
                            # littlefs-python defaults to (2,1) which the firmware rejects as corrupt.


def build_image(out_path, files):
    """files: iterable of (local_path, remote_name) tuples."""
    fs = LittleFS(
        block_size=BLOCK_SIZE,
        block_count=BLOCK_COUNT,
        read_size=READ_SIZE,
        prog_size=PROG_SIZE,
        lookahead_size=LOOKAHEAD,
        block_cycles=BLOCK_CYCLES,
        disk_version=DISK_VERSION,
    )
    total_bytes = 0
    for local, remote in files:
        data = Path(local).read_bytes()
        with fs.open(remote, 'wb') as f:
            f.write(data)
        total_bytes += len(data)
        print('  + {:>30}  {:>10} B  -> /{}'.format(
            os.path.basename(local), len(data), remote))
    Path(out_path).write_bytes(bytes(fs.context.buffer))
    print('  built %s (%d B raw image, %d B payload)' % (
        out_path, len(fs.context.buffer), total_bytes))


def esptool(args):
    cmd = [sys.executable, '-m', 'esptool'] + args
    print('  $', ' '.join(cmd))
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.time() - t0
    if r.returncode != 0:
        sys.stderr.write(r.stdout); sys.stderr.write(r.stderr)
        raise SystemExit('esptool failed (rc=%d) after %.1fs' % (r.returncode, dt))
    return r.stdout, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('files', nargs='+',
                    help='local files to include; first is renamed via --remote-name')
    ap.add_argument('--port', default='COM12')
    ap.add_argument('--baud', type=int, default=460800)
    ap.add_argument('--remote-name', default='model.tflite',
                    help='filename for the FIRST positional file on the device')
    ap.add_argument('--no-backup', action='store_true')
    ap.add_argument('--dry-run', action='store_true',
                    help='build the image but do not flash')
    args = ap.parse_args()

    out_dir = Path(__file__).parent
    new_img = out_dir / '_vfs_new.bin'
    backup = out_dir / '_vfs_backup.bin'

    # Map files: first file gets --remote-name, others keep their basename.
    file_map = []
    for i, f in enumerate(args.files):
        remote = args.remote_name if i == 0 else os.path.basename(f)
        file_map.append((f, remote))

    print('Building LittleFS image ({} blocks * {} B = {} B):'.format(
        BLOCK_COUNT, BLOCK_SIZE, VFS_SIZE))
    build_image(str(new_img), file_map)

    if args.dry_run:
        print('Dry run: skipping backup + flash.')
        return

    if not args.no_backup:
        print('\nBacking up existing VFS partition (0x{:x}, {} B) ...'.format(
            VFS_OFFSET, VFS_SIZE))
        _, dt = esptool([
            '--chip', 'esp32s3', '--port', args.port, '--baud', str(args.baud),
            '--before', 'default-reset', '--after', 'no-reset',
            'read-flash', hex(VFS_OFFSET), hex(VFS_SIZE), str(backup),
        ])
        rate = VFS_SIZE / dt / 1024
        print('  read in {:.1f}s  ({:.0f} KB/s)'.format(dt, rate))

    print('\nWriting new VFS image ...')
    _, dt = esptool([
        '--chip', 'esp32s3', '--port', args.port, '--baud', str(args.baud),
        '--before', 'default-reset', '--after', 'hard-reset',
        'write-flash', hex(VFS_OFFSET), str(new_img),
    ])
    rate = VFS_SIZE / dt / 1024
    print('  wrote 6 MB in {:.1f}s  ({:.0f} KB/s wall, including erase)'.format(
        dt, rate))

    payload = sum(Path(f).stat().st_size for f, _ in file_map)
    payload_rate = payload / dt / 1024
    print('  payload {} B effective rate {:.0f} KB/s'.format(payload, payload_rate))


if __name__ == '__main__':
    main()
