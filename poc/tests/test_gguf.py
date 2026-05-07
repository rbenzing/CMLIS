"""Tests for cmlis/gguf.py."""

from __future__ import annotations

import struct
import tempfile
from pathlib import Path

import pytest

from cmlis.gguf import GGUFError, GGUFMetadata, read_gguf_metadata, validate_gguf_path

# ---------------------------------------------------------------------------
# Helpers to build minimal valid GGUF byte sequences
# ---------------------------------------------------------------------------

MAGIC = b"GGUF"


def _make_gguf_v3(
    tensor_count: int = 0,
    kv_pairs: list[tuple[str, int, object]] | None = None,
) -> bytes:
    """Build a minimal GGUF v3 byte sequence with optional metadata KV pairs."""
    kv_pairs = kv_pairs or []
    buf = bytearray()
    buf += MAGIC
    buf += struct.pack("<I", 3)  # version
    buf += struct.pack("<Q", tensor_count)  # tensor_count
    buf += struct.pack("<Q", len(kv_pairs))  # kv_count

    for key, vtype, value in kv_pairs:
        # write key (uint64 length + utf-8 bytes)
        kb = key.encode("utf-8")
        buf += struct.pack("<Q", len(kb))
        buf += kb
        # write value type
        buf += struct.pack("<I", vtype)
        # write value payload
        if vtype == 4:  # UINT32
            buf += struct.pack("<I", value)
        elif vtype == 10:  # UINT64
            buf += struct.pack("<Q", value)
        elif vtype == 8:  # STRING
            vb = value.encode("utf-8")
            buf += struct.pack("<Q", len(vb))
            buf += vb

    return bytes(buf)


def _write_tmp(data: bytes) -> str:
    f = tempfile.NamedTemporaryFile(delete=False, suffix=".gguf")
    f.write(data)
    f.close()
    return f.name


# ---------------------------------------------------------------------------
# validate_gguf_path
# ---------------------------------------------------------------------------


def test_validate_missing_file():
    with pytest.raises(GGUFError, match="file not found"):
        validate_gguf_path("/nonexistent/path/model.gguf")


def test_validate_bad_magic(tmp_path):
    p = tmp_path / "bad.gguf"
    p.write_bytes(b"JUNK" + struct.pack("<I", 3))
    with pytest.raises(GGUFError, match="not a GGUF file"):
        validate_gguf_path(str(p))


def test_validate_file_too_small(tmp_path):
    p = tmp_path / "tiny.gguf"
    p.write_bytes(b"GGU")  # 3 bytes, < 8
    with pytest.raises(GGUFError, match="too small"):
        validate_gguf_path(str(p))


def test_validate_unsupported_version(tmp_path):
    p = tmp_path / "v9.gguf"
    p.write_bytes(MAGIC + struct.pack("<I", 9))
    with pytest.raises(GGUFError, match="unsupported GGUF version"):
        validate_gguf_path(str(p))


def test_validate_v3_ok(tmp_path):
    p = tmp_path / "valid.gguf"
    p.write_bytes(_make_gguf_v3())
    validate_gguf_path(str(p))  # must not raise


def test_validate_v2_ok(tmp_path):
    p = tmp_path / "v2.gguf"
    # v2 uses same magic, version=2, uint64 tensor/kv counts
    buf = MAGIC + struct.pack("<I", 2) + struct.pack("<Q", 0) + struct.pack("<Q", 0)
    p.write_bytes(buf)
    validate_gguf_path(str(p))


def test_validate_v1_ok(tmp_path):
    p = tmp_path / "v1.gguf"
    # v1 uses uint32 tensor/kv counts
    buf = MAGIC + struct.pack("<I", 1) + struct.pack("<I", 0) + struct.pack("<I", 0)
    p.write_bytes(buf)
    validate_gguf_path(str(p))


# ---------------------------------------------------------------------------
# read_gguf_metadata
# ---------------------------------------------------------------------------


def test_read_metadata_minimal(tmp_path):
    p = tmp_path / "minimal.gguf"
    p.write_bytes(_make_gguf_v3(tensor_count=128))
    meta = read_gguf_metadata(str(p))
    assert isinstance(meta, GGUFMetadata)
    assert meta.version == 3
    assert meta.tensor_count == 128
    assert meta.quant_type is None
    assert meta.model_arch is None


def test_read_metadata_with_arch(tmp_path):
    p = tmp_path / "arch.gguf"
    p.write_bytes(_make_gguf_v3(kv_pairs=[("general.architecture", 8, "llama")]))
    meta = read_gguf_metadata(str(p))
    assert meta.model_arch == "llama"


def test_read_metadata_with_file_type_quant(tmp_path):
    # file_type=17 → Q5_K_M
    p = tmp_path / "quant.gguf"
    p.write_bytes(_make_gguf_v3(kv_pairs=[("general.file_type", 4, 17)]))
    meta = read_gguf_metadata(str(p))
    assert meta.quant_type == "Q5_K_M"


def test_read_metadata_estimated_ram_fallback(tmp_path):
    # No quant or param info — should fall back to file-size * 1.2
    p = tmp_path / "noram.gguf"
    data = _make_gguf_v3()
    p.write_bytes(data)
    meta = read_gguf_metadata(str(p))
    file_mb = p.stat().st_size // (1024 * 1024)
    assert meta.estimated_ram_mb == int(file_mb * 1.2)


def test_read_metadata_invalid_raises(tmp_path):
    p = tmp_path / "bad.gguf"
    p.write_bytes(b"JUNK" + struct.pack("<I", 3))
    with pytest.raises(GGUFError):
        read_gguf_metadata(str(p))


def test_read_metadata_multiple_kv(tmp_path):
    kv = [
        ("general.architecture", 8, "mistral"),
        ("general.file_type", 4, 15),  # Q4_K_M
    ]
    p = tmp_path / "multi.gguf"
    p.write_bytes(_make_gguf_v3(tensor_count=256, kv_pairs=kv))
    meta = read_gguf_metadata(str(p))
    assert meta.model_arch == "mistral"
    assert meta.quant_type == "Q4_K_M"
    assert meta.tensor_count == 256
