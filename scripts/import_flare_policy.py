#!/usr/bin/env python3
"""Import a trusted rsl_rl v1.0.2 FLARE checkpoint for inference on OpenCL."""
import argparse
import hashlib
import json
import os
import pickle
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import yaml
from tinygrad import Context, Device, Tensor
from tinygrad.nn.state import get_state_dict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flare_q6a.runners.ppo import ActorCritic, load_policy
from flare_q6a.utils.config import FLEX_TASK, environment_config, task_signature
from flare_q6a.utils.hardware import require_adreno


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True, help='trusted original FLARE model_*.pt')
    parser.add_argument('--config', type=Path, help='trusted saved FLARE config.yaml or cfgs.pkl (defaults to checkpoint directory)')
    parser.add_argument('--output', type=Path, required=True, help='new inference-only .npz path')
    parser.add_argument('--allow-unverified-config', action='store_true', help='allow conversion without saved environment config')
    args = parser.parse_args()
    source = args.checkpoint.resolve(strict=True)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f'Output already exists: {output}')
    config = args.config.resolve(strict=True) if args.config else next((source.parent / n for n in ('config.yaml', 'cfgs.pkl') if (source.parent / n).is_file()), source.parent / 'config.yaml')
    config_verified = config.is_file()
    if not config_verified and not args.allow_unverified_config:
        parser.error('saved config.yaml not found; supply --config or explicitly opt in with --allow-unverified-config')
    if config_verified:
        if config.suffix == '.pkl':
            # FLARE itself writes cfgs.pkl; only load files from a trusted source.
            with config.open('rb') as file:
                blocks = pickle.load(file)
            if not isinstance(blocks, (list, tuple)) or len(blocks) != 5:
                raise ValueError('invalid original FLARE cfgs.pkl')
            saved = dict(zip(('env_cfg', 'obs_cfg', 'reward_cfg', 'command_cfg', 'train_cfg'), blocks))
        else:
            saved = yaml.safe_load(config.read_text())
        original = environment_config(False)
        for key in ('env_cfg', 'obs_cfg', 'command_cfg', 'reward_cfg'):
            if saved.get(key) != original[key]:
                raise ValueError(f'{key} differs from the local FLARE flexible configuration')
        policy = saved.get('train_cfg', {}).get('policy', {})
        if policy.get('activation') != 'tanh' or policy.get('actor_hidden_dims') != [128, 128] or policy.get('critic_hidden_dims') != [128, 128]:
            raise ValueError('saved policy architecture differs from 26-128-128-4/1 tanh')
    hardware = require_adreno()
    if Device.DEFAULT != 'CL':
        raise RuntimeError('OpenCL GPU required')
    # weights_only=True rejects untrusted arbitrary pickle globals; accept trusted sources only.
    checkpoint = torch.load(source, map_location='cpu', weights_only=True)
    state = checkpoint['model_state_dict']
    arrays = {}
    for network in ('actor', 'critic'):
        for target, upstream in enumerate((0, 2, 4)):
            for name in ('weight', 'bias'):
                tensor = state[f'{network}.{upstream}.{name}']
                if tensor.dtype != torch.float32:
                    raise ValueError(f'{network}.{upstream}.{name} must be FP32')
                arrays[f'model.{network}.{target}.{name}'] = tensor.detach().cpu().numpy().copy()
    arrays['model.std'] = state['std'].detach().cpu().numpy().copy()
    if arrays['model.std'].dtype != np.float32 or not np.isfinite(arrays['model.std']).all() or (arrays['model.std'] <= 0).any():
        raise ValueError('checkpoint policy standard deviation must be finite, positive FP32')
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise ValueError('checkpoint contains non-finite model weights')
    meta = {'schema': 'flare-inference-v1', 'iteration': int(checkpoint['iter']),
            'source_sha256': digest(source), 'saved_config_sha256': digest(config) if config_verified else None,
            'environment_config_verified': config_verified, 'optimizer_included': False,
            'run': {'experiment': FLEX_TASK, 'physics': 'genesis', 'task_signature': task_signature(False)}}
    arrays['metadata'] = np.array(json.dumps(meta))
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_name = None
    try:
        with tempfile.NamedTemporaryFile(prefix='.policy-', suffix='.npz', dir=output.parent, delete=False) as tmp:
            tmp_name = tmp.name
            np.savez(tmp, **arrays)
        model = ActorCritic(26)
        load_policy(model, tmp_name)
        torch.set_num_threads(2)
        obs = np.random.default_rng(17).uniform(-1.5, 1.5, (64, 26)).astype(np.float32)
        def reference(network):
            value = torch.from_numpy(obs)
            for layer in (0, 2, 4):
                value = torch.nn.functional.linear(value, state[f'{network}.{layer}.weight'], state[f'{network}.{layer}.bias'])
                if layer != 4:
                    value = torch.tanh(value)
            return value.detach().cpu().numpy()
        with torch.no_grad(), Context(TRAINING=0):
            mean, value = model.forward(Tensor(obs, device='CL'))
            actual = {'actor': mean.numpy(), 'critic': value.numpy()}
            expected = {'actor': reference('actor'), 'critic': reference('critic').reshape(-1)}
        errors = {}
        for name in ('actor', 'critic'):
            errors[name] = float(np.max(np.abs(actual[name] - expected[name])))
            np.testing.assert_allclose(actual[name], expected[name], rtol=1e-4, atol=2e-5)
        for name, tensor in get_state_dict(model).items():
            expected_array = arrays['model.' + name]
            np.testing.assert_array_equal(tensor.numpy(), expected_array)
        os.link(tmp_name, output)  # fail if output appeared meanwhile; never replace user files
        print(json.dumps({'status': 'PASS', 'checkpoint': str(output), 'iteration': meta['iteration'],
                          'source_sha256': meta['source_sha256'], 'output_sha256': digest(output),
                          'config_verified': config_verified, 'gpu': hardware,
                          'reference_test': 'PyTorch CPU is used only for conversion verification',
                          'max_absolute_error': errors, 'resume_training': False}, indent=2))
    finally:
        if tmp_name:
            Path(tmp_name).unlink(missing_ok=True)


if __name__ == '__main__':
    main()
