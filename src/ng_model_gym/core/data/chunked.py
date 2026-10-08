# SPDX-FileCopyrightText: Copyright 2026 Indie-Siggi
# SPDX-License-Identifier: Apache-2.0
"""Safetensors captures split into per-frame zstd chunks (.stz).

A capture holds every tensor with the frame as its first dimension. Training reads short
windows of frames, so a whole-file compression would decompress the full capture for every
window; here each frame's tensors are one safetensors blob compressed with zstd, behind an
index, and a window decompresses only its own frames. Lossless: a window read from the .stz
file equals the same slice of the source .safetensors file.

Layout: b"STZ1", the header length (uint64, little endian), a JSON header {"metadata",
"frames", "keys", "chunks": [[offset, size], ...]} with offsets from the end of the header,
then the chunks.
"""

import json
import struct
from pathlib import Path

import numpy as np
import safetensors
import safetensors.numpy
import safetensors.torch
import torch
import zstandard

MAGIC = b"STZ1"
EXTENSION = ".stz"


def write_stz(src: Path, dst: Path, level: int = 3) -> None:
    """Writes the .safetensors capture `src` as the per-frame chunked capture `dst`."""
    with safetensors.safe_open(src, framework="numpy") as f:
        metadata = f.metadata() or {}
        keys = list(f.keys())
        frames = (
            int(metadata["Length"])
            if "Length" in metadata
            else f.get_slice(keys[0]).get_shape()[0]
        )
        for k in keys:
            if f.get_slice(k).get_shape()[0] != frames:
                raise ValueError(
                    f"{src}: {k} does not have the frame as its first dimension"
                )
        cz = zstandard.ZstdCompressor(level=level)
        chunks, offset = [], 0
        tmp = dst.with_name(dst.name + ".part")
        with open(tmp, "wb") as body:
            for i in range(frames):
                blob = cz.compress(
                    safetensors.numpy.save({k: f.get_slice(k)[i : i + 1] for k in keys})
                )
                body.write(blob)
                chunks.append([offset, len(blob)])
                offset += len(blob)
    header = json.dumps(
        {"metadata": metadata, "frames": frames, "keys": keys, "chunks": chunks}
    ).encode()
    with open(dst, "wb") as out, open(tmp, "rb") as body:
        out.write(MAGIC + struct.pack("<Q", len(header)) + header)
        while block := body.read(1 << 24):
            out.write(block)
    tmp.unlink()


class _Slice:
    """One tensor of an StzFile, sliced along the frames like a safetensors slice."""

    def __init__(self, owner, key):
        self.owner, self.key = owner, key

    def get_shape(self):
        """The tensor's full shape, frames first."""
        return [self.owner.frames, *self.owner.frame(0)[self.key].shape[1:]]

    def __getitem__(self, item):
        if not isinstance(item, slice):
            item = slice(item, item + 1) if item >= 0 else slice(item, item + 1 or None)
            return self[item][0]
        idx = range(*item.indices(self.owner.frames))
        parts = [self.owner.frame(i)[self.key] for i in idx]
        if self.owner.framework == "pt":
            return torch.cat(parts) if parts else torch.empty(0)
        return np.concatenate(parts) if parts else np.empty(0)


class StzFile:
    """Read access to a .stz capture with the subset of safetensors' safe_open interface the
    datasets use: metadata(), keys(), get_slice(key)[start:stop] and get_tensor(key). Decoded
    frames are kept while it is open."""

    def __init__(self, path: Path, framework: str = "pt"):
        self.path, self.framework = Path(path), framework
        self.file = open(self.path, "rb")  # pylint: disable=consider-using-with
        if self.file.read(4) != MAGIC:
            raise ValueError(f"{path}: not a chunked safetensors capture")
        (n,) = struct.unpack("<Q", self.file.read(8))
        header = json.loads(self.file.read(n))
        self.base = 12 + n
        self._metadata, self.frames = header["metadata"], header["frames"]
        self._keys, self.chunks = header["keys"], header["chunks"]
        self.cache, self.dz = {}, zstandard.ZstdDecompressor()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.file.close()

    def metadata(self):
        """The source capture's metadata."""
        return self._metadata

    def keys(self):
        """The tensor names."""
        return list(self._keys)

    def frame(self, i: int) -> dict:
        """Frame i's tensors (first dimension 1), decoded once."""
        if i not in self.cache:
            offset, size = self.chunks[i]
            self.file.seek(self.base + offset)
            blob = self.dz.decompress(self.file.read(size))
            load = (
                safetensors.torch.load
                if self.framework == "pt"
                else safetensors.numpy.load
            )
            self.cache[i] = load(blob)
        return self.cache[i]

    def get_slice(self, key: str) -> _Slice:
        """A tensor to slice along the frames."""
        if key not in self._keys:
            raise KeyError(key)
        return _Slice(self, key)

    def get_tensor(self, key: str):
        """A whole tensor, all frames."""
        return self.get_slice(key)[:]


def open_capture(path: Path, framework: str = "pt"):
    """safetensors.safe_open for .safetensors files, StzFile for .stz files."""
    if Path(path).suffix == EXTENSION:
        return StzFile(path, framework)
    return safetensors.safe_open(path, framework=framework, device="cpu")
