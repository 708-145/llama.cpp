import struct
import numpy as np
import pytest

from gguf.constants import GGMLQuantizationType
from gguf.gguf_reader import GGUFReader


def _write_gguf(path, n_dims_field, dims):
    buf = b'GGUF' + struct.pack('<IQQ', 3, 1, 0)  # version 3, 1 tensor, 0 kv
    name = b'bad_tensor'
    buf += struct.pack('<Q', len(name)) + name
    buf += struct.pack('<I', n_dims_field)
    for d in dims:
        buf += struct.pack('<Q', d)
    buf += struct.pack('<I', 0)  # dtype F32
    buf += struct.pack('<Q', 0)  # tensor offset
    buf += b'\x00' * 64
    path.write_bytes(buf)


def test_n_dims_upper_bound(tmp_path):
    # crafted file claims 1_000_000 dims; must be rejected, not read past EOF
    p = tmp_path / 'evil_ndims.gguf'
    _write_gguf(p, 1_000_000, [1] * 8)
    with pytest.raises(ValueError, match='exceeds GGML_MAX_DIMS'):
        GGUFReader(p)


def test_dims_product_no_uint64_wraparound(tmp_path):
    # dims whose true product overflows uint64; np.prod would wrap to 4 and
    # silently pass an undersized read. The reader must not accept it.
    dims = [4194305, 4194305, 211106198978564]
    assert int(np.prod(np.array(dims, dtype=np.uint64))) == 4  # the wrap bug
    p = tmp_path / 'evil_overflow.gguf'
    _write_gguf(p, len(dims), dims)
    with pytest.raises(ValueError):
        GGUFReader(p)


def test_iq3_cxs2_row_footer_layout(tmp_path):
    p = tmp_path / 'iq3_cxs2.gguf'
    buf = b'GGUF' + struct.pack('<IQQ', 3, 1, 0)
    name = b'iq3_cxs2_tensor'
    buf += struct.pack('<Q', len(name)) + name
    buf += struct.pack('<I', 2)
    buf += struct.pack('<QQ', 1536, 2)
    buf += struct.pack('<I', GGMLQuantizationType.IQ3_CXS2)
    buf += struct.pack('<Q', 0)
    buf += b'\x00' * 4096
    p.write_bytes(buf)

    tensor = GGUFReader(p).tensors[0]
    assert tensor.n_bytes == 2 * (10 * 64 + 8)
    assert tensor.data.shape == (2, 10 * 64 + 8)
