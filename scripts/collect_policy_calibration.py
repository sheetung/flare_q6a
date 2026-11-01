#!/usr/bin/env python3
"""Collect actual FLARE simulator observations for QAIRT calibration."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from tinygrad import Context, Device, Tensor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flare_q6a.runners.ppo import ActorCritic, load_policy
from flare_q6a.utils.config import FLEX_TASK, task_signature
from flare_q6a.utils.hardware import require_adreno


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--envs', type=int, default=4)
    p.add_argument('--steps', type=int, default=500)
    p.add_argument('--seed', type=int, default=123)
    a = p.parse_args()
    if a.envs < 1 or a.steps < 1 or a.seed < 0:
        p.error('envs and steps must be positive, seed must be non-negative')
    source = a.checkpoint.resolve(strict=True)
    output = a.output_dir.resolve()
    with np.load(source, allow_pickle=False) as data:
        if 'metadata' not in data:
            raise ValueError('checkpoint must include task metadata')
        meta = json.loads(str(data['metadata']))
    run = meta.get('run', {})
    if run.get('experiment') != FLEX_TASK or run.get('task_signature') != task_signature(False):
        raise ValueError('expected a matching flexible FLARE waypoint policy')
    require_adreno()
    if Device.DEFAULT != 'CL':
        raise RuntimeError('OpenCL GPU required')
    output.mkdir(parents=True, exist_ok=False)
    np.random.seed(a.seed)
    Tensor.manual_seed(a.seed)
    model = ActorCritic(26)
    load_policy(model, source)
    from flare_q6a.envs.genesis_bridge import GenesisBridge, torch
    torch.manual_seed(a.seed)
    env = GenesisBridge(a.envs)
    obs = env.obs()
    with (output / 'input_list.txt').open('w') as inputs, Context(TRAINING=0):
        for step in range(a.steps):
            values = np.asarray(obs.numpy(), dtype='<f4')
            if values.shape != (a.envs, 26) or not np.isfinite(values).all():
                raise ValueError('invalid simulator observation')
            for i, row in enumerate(values):
                path = output / f'{step:06d}_{i:04d}.raw'
                row.tofile(path)
                inputs.write(str(path) + '\n')
            mean, _ = model.forward(obs)
            obs, _, _, _, _ = env.step(mean.detach().realize())
            if (step + 1) % 50 == 0:
                print(f'Collected {step + 1}/{a.steps} steps', flush=True)
    manifest = {'checkpoint_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                'task': FLEX_TASK, 'seed': a.seed, 'envs': a.envs, 'steps': a.steps,
                'samples': a.envs * a.steps, 'input_shape': [1, 26], 'dtype': 'float32',
                'note': 'Simulation samples only; assess representativeness for deployment.'}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Calibration samples: {manifest["samples"]}; input list: {output / "input_list.txt"}')


if __name__ == '__main__':
    main()
