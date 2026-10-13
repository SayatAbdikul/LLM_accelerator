#!/usr/bin/env python3
"""New LN quality vectors; never modifies the frozen gen2 fixture corpus.

Expected values use the frozen simulator's NumPy mean/var expressions directly,
not a copy of the RTL reduction traversal. Outputs live in ignored build/.
An optional retained prefix contributes its real row only after all four array
hashes and its expected output against the unchanged expression are checked.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct

import numpy as np


def expected(x, gamma, beta):
    x = np.asarray(x, np.float16).astype(np.float32).reshape(1, -1)
    gamma = np.asarray(gamma, np.float16).astype(np.float32)
    beta = np.asarray(beta, np.float16).astype(np.float32)
    mean = x.mean(axis=-1, keepdims=True).astype(np.float32)
    var = x.var(axis=-1, keepdims=True).astype(np.float32)
    norm = (x - mean) / np.sqrt(var + np.float32(1e-5))
    return (norm * gamma + beta).astype(np.float32).astype(np.float16).reshape(-1)


def cases():
    rng = np.random.default_rng(0x4C4E)
    for width in range(16, 1025, 16):
        gamma = rng.normal(1, .2, width).astype(np.float16)
        beta = rng.normal(0, .01, width).astype(np.float16)
        vectors = {
            'normal': rng.normal(0, .03, width),
            'near_constant': 1 + rng.integers(-2, 3, width) / 1024,
            'cancellation': np.resize([4096., .001, -4096., -.0003, 1., -.1, -1., .09], width),
            'wide': rng.uniform(-1, 1, width) * np.exp2(rng.integers(-20, 15, width)),
            'extrema': np.resize([65504., -65504., 0., 2**-24], width),
            'positive_zero': np.zeros(width),
            'negative_zero': np.full(width, -0.0),
            'subnormal': np.resize([2**-24, -2**-24, 2**-20, -2**-20, 0., -0.0], width),
        }
        for name, values in vectors.items():
            x = np.asarray(values, np.float16)
            g, b = gamma, beta
            if name.endswith('zero'):
                g = np.resize(np.array([1., -1.], np.float16), width)
                b = np.full(width, -0.0, np.float16)
            yield f'{name}_{width}', x, g, b, expected(x, g, b)


def captured_case(directory):
    directory = Path(directory)
    arrays = json.loads((directory / 'arrays.json').read_text())
    values = []
    for name in ('input', 'gamma', 'beta', 'golden_output'):
        item = arrays[name]
        if (item['file'] != name + '.raw' or item['bytes'] != 1536
                or item['dtype'] != 'float16' or item['shape'] != [1, 768]):
            raise ValueError('invalid captured LN array contract')
        path = directory / item['file']
        if path.is_symlink():
            raise ValueError('captured array must not be a symlink')
        raw = path.read_bytes()
        if len(raw) != 1536 or hashlib.sha256(raw).hexdigest() != item['sha256']:
            raise ValueError('captured LN array hash mismatch')
        values.append(np.frombuffer(raw, dtype='<f2').copy())
    if expected(*values[:3]).tobytes() != values[3].tobytes():
        raise ValueError('captured golden output does not match frozen NumPy expression')
    return ('captured_gpt2_block0_ln1_768', *values)


def write_vectors(path, vectors):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as f:
        f.write(b'LNPAIR01' + struct.pack('<I', len(vectors)))
        for name, x, gamma, beta, out in vectors:
            n = len(x)
            if not (16 <= n <= 1024 and n % 16 == 0):
                raise ValueError('illegal LN width')
            encoded = name.encode('ascii')
            f.write(struct.pack('<II', n, len(encoded)) + encoded)
            for array in (x, gamma, beta, out):
                if len(array) != n or not np.isfinite(array).all():
                    raise ValueError('nonfinite or wrong-sized LN array')
                f.write(np.asarray(array, dtype='<f2').tobytes())
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--captured-prefix', type=Path)
    args = parser.parse_args()
    vectors = list(cases())
    if args.captured_prefix is not None:
        vectors.append(captured_case(args.captured_prefix))
    digest = write_vectors(args.out, vectors)
    print(json.dumps({'cases': len(vectors), 'numpy': np.__version__,
                      'sha256': digest, 'out': str(args.out)}, sort_keys=True))


if __name__ == '__main__':
    main()
