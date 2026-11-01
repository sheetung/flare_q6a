#!/usr/bin/env python3
"""Import a trusted server S2 checkpoint and verify its Actor on Adreno."""
import argparse, hashlib, json, pickle, sys
from pathlib import Path
import numpy as np
import torch
import yaml
from tinygrad import Context, Tensor, nn
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flare_q6a.utils.hardware import require_adreno
from flare_q6a.envs.server_s2_bridge import server_s2_signature

class Actor:
    def __init__(self, arrays):
        self.layers=[]
        for i in range(4):
            weight=arrays[f'weight_{i}']; bias=arrays[f'bias_{i}']
            layer=nn.Linear(weight.shape[1], weight.shape[0])
            layer.weight=Tensor(weight.copy(),device='CL').realize()
            layer.bias=Tensor(bias.copy(),device='CL').realize()
            self.layers.append(layer)
    def __call__(self, obs):
        for layer in self.layers[:-1]: obs=layer(obs).elu()
        return self.layers[-1](obs)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True,help='trusted original cfgs.pkl')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); source=a.checkpoint.resolve(strict=True); config=a.config.resolve(strict=True); out=a.output.resolve()
    if out.exists(): raise FileExistsError(out)
    with config.open('rb') as f: cfg=pickle.load(f)
    if not isinstance(cfg,(list,tuple)) or len(cfg)!=5: raise ValueError('invalid cfgs.pkl')
    expected=yaml.safe_load((ROOT/'flare_q6a/config/envs/server_s2_rigid.yaml').read_text())
    for i,key in enumerate(('env_cfg','obs_cfg','reward_cfg','command_cfg')):
        if cfg[i] != expected[key]: raise ValueError(f'S2 configuration differs: {key}')
    policy=cfg[4]['policy']
    if policy.get('actor_hidden_dims') != [256,256,128] or policy.get('activation') != 'elu': raise ValueError('S2 actor architecture differs')
    hardware=require_adreno(); checkpoint=torch.load(source,map_location='cpu',weights_only=True); state=checkpoint['model_state_dict']
    arrays={}
    for i,k in enumerate((0,2,4,6)):
        for kind in ('weight','bias'):
            v=state[f'actor.{k}.{kind}'].detach().cpu().numpy()
            if v.dtype!=np.float32 or not np.isfinite(v).all(): raise ValueError('invalid actor weights')
            arrays[f'{kind}_{i}']=v.copy()
    if [tuple(arrays[f'weight_{i}'].shape) for i in range(4)] != [(256,26),(256,256),(128,256),(4,128)]: raise ValueError('wrong actor shapes')
    model=Actor(arrays); observations=np.random.default_rng(19).uniform(-1,1,(64,26)).astype(np.float32)
    with torch.no_grad(), Context(TRAINING=0):
        ref=torch.from_numpy(observations)
        for i,k in enumerate((0,2,4,6)):
            ref=torch.nn.functional.linear(ref,state[f'actor.{k}.weight'],state[f'actor.{k}.bias'])
            if i<3: ref=torch.nn.functional.elu(ref)
        actual=model(Tensor(observations,device='CL')).numpy(); expected_output=ref.numpy()
    np.testing.assert_allclose(actual,expected_output,rtol=1e-4,atol=2e-5)
    meta={'schema':'server-s2-actor-v1','iteration':int(checkpoint['iter']),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
          'config_sha256':hashlib.sha256(config.read_bytes()).hexdigest(),
          'urdf_sha256':hashlib.sha256((ROOT/'flare_q6a/assets/server_s2/real_sus_exp_rigid.urdf').read_bytes()).hexdigest(),
          'actor_max_abs_error':float(np.max(np.abs(actual-expected_output))), 'hardware':hardware,
          'task_signature':server_s2_signature()}
    arrays['metadata']=np.array(json.dumps(meta)); out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('xb') as f: np.savez(f,**arrays)
    print(json.dumps(meta,indent=2))
if __name__=='__main__': main()
