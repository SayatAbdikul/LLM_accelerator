"""Bounded generator/ordered-reduction checks; no RTL builds or model loads."""
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

from gen_ln_vectors import captured_case, cases, expected, write_vectors


def iterative_sum(array):
    """Small model of the bounded, nonrecursive hardware traversal."""
    array = np.asarray(array, np.float32)
    first, count = 0, len(array)
    stack = []
    need_leaf = True
    max_depth = 0
    while True:
        if need_leaf:
            while count > 128:
                half = count // 2 // 8 * 8
                stack.append([first + half, count - half, None])
                max_depth = max(max_depth, len(stack))
                count = half
            assert count % 8 == 0 and count >= 8
            partial = array[first:first + 8].copy()
            for i in range(8, count):
                partial[i % 8] = np.float32(partial[i % 8] + array[first + i])
            for a, b in ((0, 1), (2, 3), (0, 2), (4, 5), (6, 7), (4, 6), (0, 4)):
                partial[a] = np.float32(partial[a] + partial[b])
            result = partial[0]
            need_leaf = False
        elif not stack:
            return np.float32(np.float32(0) + result), max_depth
        elif stack[-1][2] is None:
            first, count, _ = stack[-1]
            stack[-1][2] = result
            need_leaf = True
        else:
            result = np.float32(stack.pop()[2] + result)


class LayerNormVectorsTest(unittest.TestCase):
    def test_every_legal_width_and_finite_exact_output(self):
        vectors = list(cases())
        self.assertEqual(len(vectors), 512)
        self.assertEqual(len({v[0] for v in vectors}), 512)
        for i, (_, x, gamma, beta, out) in enumerate(vectors):
            self.assertEqual(len(x), 16 * (i // 8 + 1))
            self.assertTrue(np.isfinite(out).all())
            self.assertEqual(out.tobytes(), expected(x, gamma, beta).tobytes())

    def test_iterative_tree_matches_numpy_sum_and_variance_bits(self):
        for name, x, _, _, _ in cases():
            x = x.astype(np.float32)
            value, depth = iterative_sum(x)
            self.assertLessEqual(depth, 3)
            self.assertEqual(value.tobytes(), x.sum(dtype=np.float32).tobytes(), name)
            mean = np.float32(value / np.float32(len(x)))
            square = ((x - mean) * (x - mean)).astype(np.float32)
            value, _ = iterative_sum(square)
            var = np.float32(value / np.float32(len(x)))
            self.assertEqual(var.tobytes(), x.var(dtype=np.float32).tobytes(), name)

    def test_signed_zero_kept_in_inputs_and_expected(self):
        _, x, _, _, output = next(v for v in cases() if v[0] == 'negative_zero_16')
        self.assertTrue(np.all(x.view(np.uint16) == 0x8000))
        self.assertEqual(set(output.view(np.uint16)), {0, 0x8000})

    def test_negative_control_rejects_old_serial_reduction(self):
        for _, x, gamma, beta, output in cases():
            xf = x.astype(np.float32)
            total = np.float32(0)
            for value in xf:
                total = np.float32(total + value)
            mean = np.float32(total / np.float32(len(x)))
            total = np.float32(0)
            for value in xf:
                diff = np.float32(value - mean)
                total = np.float32(total + np.float32(diff * diff))
            var = np.float32(total / np.float32(len(x)))
            norm = (xf - mean) / np.sqrt(var + np.float32(1e-5))
            old = (norm * gamma.astype(np.float32) + beta.astype(np.float32)).astype(np.float16)
            if old.tobytes() != output.tobytes():
                return
        self.fail('quality vectors failed to distinguish the old serial reduction')

    def test_binary_deterministic(self):
        with tempfile.TemporaryDirectory() as temp:
            a, b = Path(temp) / 'a.bin', Path(temp) / 'b.bin'
            self.assertEqual(write_vectors(a, list(cases())), write_vectors(b, list(cases())))
            self.assertEqual(a.read_bytes()[:12], b'LNPAIR01' + struct.pack('<I', 512))

    def test_captured_array_hash_and_expected_guard(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            arrays = {}
            x = np.arange(768, dtype=np.float16) / np.float16(32)
            gamma, beta = np.ones(768, np.float16), np.zeros(768, np.float16)
            for name, value in zip(('input', 'gamma', 'beta', 'golden_output'),
                                   (x, gamma, beta, expected(x, gamma, beta))):
                raw = value.astype('<f2').tobytes()
                (directory / (name + '.raw')).write_bytes(raw)
                arrays[name] = dict(file=name + '.raw', bytes=len(raw), dtype='float16',
                                    shape=[1, 768], sha256=hashlib.sha256(raw).hexdigest())
            (directory / 'arrays.json').write_text(json.dumps(arrays))
            self.assertEqual(captured_case(directory)[0], 'captured_gpt2_block0_ln1_768')
            (directory / 'input.raw').write_bytes(bytes(1536))
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                captured_case(directory)


if __name__ == '__main__':
    unittest.main()
