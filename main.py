"""Entrypoint for make.fun news-image-20261126: validate and emit the 26 November 2026 world-news cover."""

from __future__ import annotations

import os
import struct
import zlib
from pathlib import Path

MAX_BYTES = 1024 * 1024
MAX_SIDE = 1024
OUTPUT_NAME = "entry.png"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def validate_png(data: bytes, source: Path) -> tuple[int, int]:
    """Verify PNG signature, chunk CRCs, 1:1 dimensions <= 1024x1024, and IEND terminator."""
    if not data:
        raise ValueError(f"Empty image file: {source}")
    if len(data) > MAX_BYTES:
        raise ValueError(f"{source} is {len(data)} bytes (limit {MAX_BYTES})")
    if not data.startswith(PNG_SIGNATURE):
        raise ValueError(f"{source} does not start with PNG signature")

    offset = 8
    width = 0
    height = 0
    saw_ihdr = False
    saw_iend = False
    total = len(data)

    while offset + 12 <= total:
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        ctype = data[offset + 4 : offset + 8]
        chunk_end = offset + 8 + length
        if chunk_end + 4 > total:
            raise ValueError(f"Truncated PNG chunk {ctype!r}")
        payload = data[offset + 8 : chunk_end]
        expected_crc = struct.unpack(">I", data[chunk_end : chunk_end + 4])[0]
        actual_crc = zlib.crc32(ctype + payload) & 0xFFFFFFFF
        if actual_crc != expected_crc:
            raise ValueError(f"PNG CRC mismatch in chunk {ctype!r}")
        if ctype == b"IHDR":
            width, height = struct.unpack(">II", payload[:8])
            saw_ihdr = True
        elif ctype == b"IEND":
            saw_iend = True
            offset = chunk_end + 4
            break
        offset = chunk_end + 4

    if not (saw_ihdr and saw_iend):
        raise ValueError("PNG missing required IHDR or IEND chunk")
    if offset != total:
        raise ValueError(f"Trailing bytes after IEND ({total - offset} bytes)")
    if width <= 0 or height <= 0 or width > MAX_SIDE or height > MAX_SIDE:
        raise ValueError(f"PNG dimensions {width}x{height} exceed {MAX_SIDE}x{MAX_SIDE}")
    if width != height:
        raise ValueError(f"PNG dimensions {width}x{height} are not 1:1 square")
    return width, height


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    source = repo_root / OUTPUT_NAME
    if not source.is_file():
        raise FileNotFoundError(f"Expected cover image at {source}")
    data = source.read_bytes()
    validate_png(data, source)

    out_dir = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / OUTPUT_NAME
    target.write_bytes(data)


if __name__ == "__main__":
    main()
