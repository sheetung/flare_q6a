#!/usr/bin/env python3
"""Export the deterministic FLARE actor as an FP32 batch-1 ONNX model."""
import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True, help='local compatible .npz checkpoint')
    parser.add_argument('--output', type=Path, required=True, help='new .onnx output path')
    parser.add_argument('--onnx-module-dir', type=Path, help='optional isolated ONNX installation directory')
    args = parser.parse_args()
    if args.onnx_module_dir:
        sys.path.insert(0, str(args.onnx_module_dir.resolve(strict=True)))
    try:
        import onnx
        from onnx.reference import ReferenceEvaluator
    except ImportError as error:
        raise SystemExit('ONNX is required; install onnx==1.16.2 or use --onnx-module-dir') from error
    source, output = args.checkpoint.resolve(strict=True), args.output.resolve()
    if output.exists():
        raise FileExistsError(f'Output already exists: {output}')
    model = torch.nn.Sequential(torch.nn.Linear(26, 128), torch.nn.Tanh(),
                                torch.nn.Linear(128, 128), torch.nn.Tanh(),
                                torch.nn.Linear(128, 4))
    with np.load(source, allow_pickle=False) as data, torch.no_grad():
        prefix = 'model.' if 'metadata' in data else ''
        for layer, src_idx in zip((0, 2, 4), (0, 1, 2)):
            for name in ('weight', 'bias'):
                array = data[f'{prefix}actor.{src_idx}.{name}']
                target = getattr(model[layer], name)
                if array.dtype != np.float32 or tuple(array.shape) != tuple(target.shape) or not np.isfinite(array).all():
                    raise ValueError(f'invalid FP32 actor tensor: {src_idx}.{name}')
                target.copy_(torch.from_numpy(array.copy()))
    model.eval()
    obs = np.random.default_rng(23).uniform(-1.5, 1.5, (1, 26)).astype(np.float32)
    with torch.no_grad():
        expected = model(torch.from_numpy(obs)).numpy()
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_name = None
    try:
        with tempfile.NamedTemporaryFile(prefix='.actor-', suffix='.onnx', dir=output.parent, delete=False) as tmp:
            tmp_name = tmp.name
        torch.onnx.export(model, torch.from_numpy(obs), tmp_name, export_params=True, opset_version=13,
                          do_constant_folding=True, input_names=['observation'], output_names=['action_mean'])
        graph = onnx.load(tmp_name)
        onnx.checker.check_model(graph)
        actual = ReferenceEvaluator(graph).run(None, {'observation': obs})[0]
        np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=1e-4)
        os.link(tmp_name, output)
        print(json.dumps({'status': 'PASS', 'onnx': str(output), 'onnx_sha256': digest(output),
                          'source_checkpoint_sha256': digest(source), 'opset': 13,
                          'input': {'name': 'observation', 'shape': [1, 26], 'dtype': 'float32'},
                          'output': {'name': 'action_mean', 'shape': [1, 4], 'dtype': 'float32'},
                          'max_absolute_error': float(np.max(np.abs(actual - expected))),
                          'scope': 'deterministic actor only; ONNX validation on CPU; NPU not yet converted or tested',
                          'action_clipping_in_graph': False}, indent=2))
    finally:
        if tmp_name:
            Path(tmp_name).unlink(missing_ok=True)


if __name__ == '__main__':
    main()
