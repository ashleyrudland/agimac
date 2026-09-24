"""Lossless transport: uint32 tokens → uint16 + gzip → original uint32 bytes."""

import gzip
from pathlib import Path
import numpy as np


def pack_tokens(source, dest):
    source, dest = Path(source), Path(dest)
    if source.stat().st_size % 4:
        raise ValueError("Token file is not aligned to uint32")
    tokens = np.memmap(source, mode="r", dtype="<u4")
    with gzip.open(dest, "wb", compresslevel=1) as out:
        for start in range(0, len(tokens), 1_000_000):
            chunk = tokens[start : start + 1_000_000]
            if chunk.max(initial=0) > 65535:
                raise ValueError("Vocabulary does not fit lossless uint16 transport")
            out.write(chunk.astype("<u2").tobytes())


def unpack_tokens(source, dest):
    with gzip.open(source, "rb") as inp, Path(dest).open("wb") as out:
        while chunk := inp.read(2_000_000):
            if len(chunk) % 2:
                raise ValueError("Truncated uint16 token stream")
            out.write(np.frombuffer(chunk, dtype="<u2").astype("<u4").tobytes())
