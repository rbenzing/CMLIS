"""GGUF file inspection.

Pure-Python reader for the GGUF binary format (v1/v2/v3).
Provides header validation and metadata extraction without running llama.cpp.

GGUF spec: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

GGUF_MAGIC = b"GGUF"

# GGUF metadata value types
_UINT8 = 0
_INT8 = 1
_UINT16 = 2
_INT16 = 3
_UINT32 = 4
_INT32 = 5
_FLOAT32 = 6
_BOOL = 7
_STRING = 8
_ARRAY = 9
_UINT64 = 10
_INT64 = 11
_FLOAT64 = 12

# Known quantization type IDs → human-readable names
# From ggml.h enum ggml_type
_QUANT_NAMES: dict[int, str] = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    6: "Q5_0",
    7: "Q5_1",
    8: "Q8_0",
    9: "Q8_1",
    10: "Q2_K",
    11: "Q3_K_S",
    12: "Q3_K_M",
    13: "Q3_K_L",
    14: "Q4_K_S",
    15: "Q4_K_M",
    16: "Q5_K_S",
    17: "Q5_K_M",
    18: "Q6_K",
    19: "Q8_K",
    20: "IQ2_XXS",
    21: "IQ2_XS",
    22: "IQ3_XXS",
    23: "IQ1_S",
    24: "IQ4_NL",
    25: "IQ3_S",
    26: "IQ2_S",
    27: "IQ4_XS",
    28: "I8",
    29: "I16",
    30: "I32",
    31: "I64",
    32: "F64",
    33: "IQ1_M",
    34: "BF16",
}

# Approximate bytes-per-weight for memory estimation (varies by quant)
_BYTES_PER_WEIGHT: dict[str, float] = {
    "F32": 4.0,
    "F16": 2.0,
    "BF16": 2.0,
    "Q8_0": 1.0,
    "Q8_1": 1.0,
    "Q8_K": 1.0,
    "Q6_K": 0.75,
    "Q5_K_M": 0.625,
    "Q5_K_S": 0.625,
    "Q5_0": 0.625,
    "Q5_1": 0.625,
    "Q4_K_M": 0.5,
    "Q4_K_S": 0.5,
    "Q4_0": 0.5,
    "Q4_1": 0.5,
    "Q3_K_L": 0.375,
    "Q3_K_M": 0.375,
    "Q3_K_S": 0.375,
    "Q3_K_XXS": 0.375,
    "Q2_K": 0.25,
    "IQ4_XS": 0.5,
    "IQ4_NL": 0.5,
    "IQ3_S": 0.375,
    "IQ3_XXS": 0.375,
    "IQ2_S": 0.25,
    "IQ2_XS": 0.25,
    "IQ2_XXS": 0.25,
    "IQ1_S": 0.125,
    "IQ1_M": 0.125,
}


@dataclass
class GGUFMetadata:
    version: int
    tensor_count: int
    quant_type: str | None
    model_arch: str | None
    param_count: int | None
    estimated_ram_mb: int | None


class GGUFError(ValueError):
    """Raised when a file fails GGUF validation."""


def validate_gguf_path(path: str) -> None:
    """Raise GGUFError if *path* is not a valid GGUF file.

    Only reads the first 8 bytes (magic + version); cheap to call before launch.
    """
    p = Path(path)
    if not p.exists():
        raise GGUFError(f"file not found: {path!r}")
    if p.stat().st_size < 8:
        raise GGUFError(f"file too small to be a valid GGUF: {path!r}")
    with p.open("rb") as f:
        header = f.read(8)
    if header[:4] != GGUF_MAGIC:
        raise GGUFError(
            f"not a GGUF file (expected magic {GGUF_MAGIC!r}, got {header[:4]!r}): {path!r}"
        )
    version = struct.unpack_from("<I", header, 4)[0]
    if version not in (1, 2, 3):
        raise GGUFError(f"unsupported GGUF version {version} in {path!r}")


def read_gguf_metadata(path: str) -> GGUFMetadata:
    """Read GGUF header metadata without loading tensors.

    Returns a GGUFMetadata dataclass. Fields that cannot be determined are None.
    Raises GGUFError on invalid files.
    """
    validate_gguf_path(path)
    with Path(path).open("rb") as f:
        data = f.read(64 * 1024)  # read first 64 KB; enough for all metadata KV pairs in practice

    offset = 4  # skip magic
    version = struct.unpack_from("<I", data, offset)[0]
    offset += 4

    if version == 1:
        tensor_count = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        kv_count = struct.unpack_from("<I", data, offset)[0]
        offset += 4
    else:
        tensor_count = struct.unpack_from("<Q", data, offset)[0]
        offset += 8
        kv_count = struct.unpack_from("<Q", data, offset)[0]
        offset += 8

    kv: dict[str, object] = {}
    try:
        for _ in range(min(kv_count, 256)):
            key, offset = _read_string(data, offset, version)
            val, offset = _read_value(data, offset, version)
            kv[key] = val
    except (struct.error, IndexError):
        pass  # partial read is fine; use what we have

    arch = kv.get("general.architecture")
    quant_id = kv.get("general.quantization_version") or kv.get("general.file_type")
    quant_name: str | None = None
    if isinstance(quant_id, int):
        quant_name = _QUANT_NAMES.get(quant_id)

    # Try to derive quant from file_type which is the most reliable key
    file_type = kv.get("general.file_type")
    if isinstance(file_type, int) and file_type in _QUANT_NAMES:
        quant_name = _QUANT_NAMES[file_type]

    param_count: int | None = None
    if isinstance(arch, str):
        for suffix in (".context_length", ".embedding_length"):
            # Param count isn't directly in KV; we use tensor_count as a proxy signal
            pass
    # Some files expose parameter count directly
    raw_params = kv.get("general.parameter_count") or kv.get("llama.parameter_count")
    if isinstance(raw_params, int):
        param_count = raw_params

    estimated_ram_mb: int | None = None
    file_size_mb = Path(path).stat().st_size // (1024 * 1024)
    if quant_name and quant_name in _BYTES_PER_WEIGHT and param_count:
        est_bytes = param_count * _BYTES_PER_WEIGHT[quant_name]
        estimated_ram_mb = int(est_bytes / (1024 * 1024)) + 512  # +512 MB overhead
    else:
        # Fall back to file size + 20% overhead
        estimated_ram_mb = int(file_size_mb * 1.2)

    return GGUFMetadata(
        version=version,
        tensor_count=int(tensor_count),
        quant_type=quant_name,
        model_arch=arch if isinstance(arch, str) else None,
        param_count=param_count,
        estimated_ram_mb=estimated_ram_mb,
    )


def _read_string(data: bytes, offset: int, version: int) -> tuple[str, int]:
    if version == 1:
        length = struct.unpack_from("<I", data, offset)[0]
        offset += 4
    else:
        length = struct.unpack_from("<Q", data, offset)[0]
        offset += 8
    s = data[offset : offset + length].decode("utf-8", errors="replace")
    return s, offset + length


def _read_value(data: bytes, offset: int, version: int) -> tuple[object, int]:
    vtype = struct.unpack_from("<I", data, offset)[0]
    offset += 4
    return _read_typed(data, offset, vtype, version)


def _read_typed(data: bytes, offset: int, vtype: int, version: int) -> tuple[object, int]:
    if vtype == _UINT8:
        return struct.unpack_from("<B", data, offset)[0], offset + 1
    if vtype == _INT8:
        return struct.unpack_from("<b", data, offset)[0], offset + 1
    if vtype == _UINT16:
        return struct.unpack_from("<H", data, offset)[0], offset + 2
    if vtype == _INT16:
        return struct.unpack_from("<h", data, offset)[0], offset + 2
    if vtype == _UINT32:
        return struct.unpack_from("<I", data, offset)[0], offset + 4
    if vtype == _INT32:
        return struct.unpack_from("<i", data, offset)[0], offset + 4
    if vtype == _FLOAT32:
        return struct.unpack_from("<f", data, offset)[0], offset + 4
    if vtype == _BOOL:
        return struct.unpack_from("<B", data, offset)[0] != 0, offset + 1
    if vtype == _STRING:
        return _read_string(data, offset, version)
    if vtype == _UINT64:
        return struct.unpack_from("<Q", data, offset)[0], offset + 8
    if vtype == _INT64:
        return struct.unpack_from("<q", data, offset)[0], offset + 8
    if vtype == _FLOAT64:
        return struct.unpack_from("<d", data, offset)[0], offset + 8
    if vtype == _ARRAY:
        item_type = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        if version == 1:
            count = struct.unpack_from("<I", data, offset)[0]
            offset += 4
        else:
            count = struct.unpack_from("<Q", data, offset)[0]
            offset += 8
        # Read array elements but only keep the first few to bound cost
        items = []
        for _ in range(min(count, 32)):
            val, offset = _read_typed(data, offset, item_type, version)
            items.append(val)
        # Skip remaining elements we didn't read
        # We can't cheaply skip variable-size elements, so we stop here
        return items, offset
    raise GGUFError(f"unknown GGUF value type {vtype}")
