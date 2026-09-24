import numpy as np
import pytest
from macoder.transfer import pack_tokens, unpack_tokens
from macoder.data import file_hash


def test_lossless_transfer(tmp_path):
    original = tmp_path / "tokens.bin"
    packed = tmp_path / "tokens.gz"
    restored = tmp_path / "restored.bin"
    np.array([0, 1, 255, 256, 16391, 65535] * 200_001, dtype="<u4").tofile(original)
    pack_tokens(original, packed)
    unpack_tokens(packed, restored)
    assert file_hash(original) == file_hash(restored)
    np.array([65536], dtype="<u4").tofile(original)
    with pytest.raises(ValueError, match="uint16"):
        pack_tokens(original, packed)
