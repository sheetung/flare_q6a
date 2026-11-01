#!/usr/bin/env python3
"""Evaluate the server's S2 rigid policy on Q6A OpenCL and Vulkan."""
import argparse, copy, hashlib, json, sys
from pathlib import Path
import numpy as np
import yaml
from tinygrad import Context, Tensor
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.import_server_s2 import Actor
from flare_q6a.utils.evaluation_metrics import EvaluationMetrics
from flare_q6a.utils.hardware import require_adreno
from flare_q6a.envs.server_s2_bridge import server_s2_signature, setup_server_s2

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--envs',type=int,default=4)
    p.add_argument('--episodes',type=int,default=4)
    p.add_argument('--steps',type=int,default=1501)
    p.add_argument('--seed',type=int,default=123)
    a=p.parse_args()
    if min(a.envs,a.episodes,a.steps)<1 or a.seed<0: p.error('envs, episodes, steps must be positive and seed nonnegative')
    source=a.checkpoint.resolve(strict=True); out=a.output.resolve()
    with np.load(source,allow_pickle=False) as data:
        meta=json.loads(str(data['metadata']))
        if meta.get('schema')=='server-s2-actor-v1':
            if meta.get('task_signature')!=server_s2_signature(): raise ValueError('S2 task signature mismatch')
            arrays={k:data[k].copy() for k in data.files if k!='metadata'}
        elif meta.get('run',{}).get('experiment')=='s2_waypoint_passing_rigid':
            if meta['run'].get('task_signature')!=server_s2_signature(): raise ValueError('S2 task signature mismatch')
            arrays={f'{name}_{i}':data[f'model.actor.{i}.{name}'].copy() for i in range(4) for name in ('weight','bias')}
        else: raise ValueError('wrong policy schema')
    asset=ROOT/'flare_q6a/assets/server_s2/real_sus_exp_rigid.urdf'
    if meta.get('urdf_sha256',server_s2_signature()['model_sha256'])!=hashlib.sha256(asset.read_bytes()).hexdigest(): raise ValueError('URDF mismatch')
    hardware=require_adreno();out.mkdir(parents=True,exist_ok=False)
    np.random.seed(a.seed);Tensor.manual_seed(a.seed);model=Actor(arrays)
    from flare_q6a.envs import genesis_bridge as bridge
    bridge.torch.manual_seed(a.seed)
    setup_server_s2()
    env=bridge.GenesisBridge(a.envs)
    obs=env.obs(); metrics=EvaluationMetrics(a.envs,env.env.dt,a.episodes)
    with (out/'episodes.jsonl').open('w') as log, Context(TRAINING=0):
        for step in range(a.steps):
            action=model(obs).detach().realize(); raw=action.numpy()
            obs,reward,done,timeout,_=env.step(action)
            if not np.isfinite(obs.numpy()).all() or not np.isfinite(raw).all(): raise ValueError('nonfinite observation/action')
            rows=metrics.observe(reward.numpy(),done.numpy(),timeout.numpy(),env.last_step_info,raw)
            for row in rows:
                log.write(json.dumps(row)+'\n');log.flush();print('Episode:',json.dumps(row),flush=True)
            if (step+1)%50==0:print(f'Step {step+1}/{a.steps}: episodes={len(metrics.rows)}',flush=True)
            if not metrics.active().any(): break
    result={'status':'PASS' if not metrics.active().any() else 'INCOMPLETE',
      'scope':'server S2 actor on Adreno; local Vulkan environment; not a flight or convergence certificate',
      'hardware':hardware,'checkpoint_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
      'steps':step+1,'seed':a.seed,'iteration':meta.get('iteration'),'task_signature':server_s2_signature(),**metrics.summary()}
    (out/'evaluation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
    return 0 if result['status']=='PASS' else 2
if __name__=='__main__':raise SystemExit(main())
