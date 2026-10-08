# SPDX-FileCopyrightText: Copyright 2026 Indie-Siggi
# SPDX-License-Identifier: Apache-2.0
import tempfile
import unittest
from pathlib import Path

import numpy as np
import safetensors
import safetensors.numpy
import torch

from ng_model_gym.core.data.chunked import open_capture, StzFile, write_stz
from ng_model_gym.core.data.data_utils import generic_safetensors_reader


class TestChunkedCapture(unittest.TestCase):
    """A .stz capture reads back exactly what its .safetensors source holds."""

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
        rng = np.random.default_rng(0)
        frames = 7
        self.tensors = {
            "colour_linear": rng.random((frames, 3, 8, 8)).astype(np.float16),
            "depth": rng.random((frames, 1, 4, 4)).astype(np.float32),
            "camera_cut": (rng.random((frames, 1)) > 0.5),
            "render_size": np.full((frames, 2), 4, np.int32),
        }
        self.src = Path(self.dir.name) / "0000.safetensors"
        self.dst = self.src.with_suffix(".stz")
        safetensors.numpy.save_file(
            self.tensors, self.src, metadata={"Length": str(frames)}
        )
        write_stz(self.src, self.dst)

    def tearDown(self):
        self.dir.cleanup()

    def test_windows_match_the_source(self):
        """Metadata, keys, shapes and every window equal the source, dtype included."""
        with (
            safetensors.safe_open(self.src, framework="pt") as a,
            open_capture(self.dst) as b,
        ):
            self.assertIsInstance(b, StzFile)
            self.assertEqual(a.metadata(), b.metadata())
            self.assertEqual(sorted(a.keys()), sorted(b.keys()))
            for k in a.keys():
                self.assertEqual(
                    list(a.get_slice(k).get_shape()), b.get_slice(k).get_shape()
                )
                for start, stop in ((0, 7), (2, 5), (6, 7)):
                    x, y = a.get_slice(k)[start:stop], b.get_slice(k)[start:stop]
                    self.assertEqual(x.dtype, y.dtype)
                    self.assertTrue(torch.equal(x, y), (k, start, stop))

    def test_generic_reader_reads_single_frames(self):
        """The frame-by-frame reader returns the same frames from either file."""
        for idx in (0, 3, 6):
            a = generic_safetensors_reader(self.src, idx)
            b = generic_safetensors_reader(self.dst, idx)
            self.assertEqual(sorted(a), sorted(b))
            for k, v in a.items():
                self.assertTrue(torch.equal(v, b[k]), (k, idx))

    def test_smaller_than_the_source(self):
        """Redundant data compresses (random noise would not, so a constant tensor here)."""
        const = {"motion": np.zeros((5, 2, 64, 64), np.float16)}
        src = Path(self.dir.name) / "flat.safetensors"
        safetensors.numpy.save_file(const, src, metadata={"Length": "5"})
        write_stz(src, src.with_suffix(".stz"))
        self.assertLess(src.with_suffix(".stz").stat().st_size, src.stat().st_size / 10)
