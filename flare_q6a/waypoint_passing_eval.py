import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

"""Frozen mean/sample policy evaluation in the GPU simulator, with saved results."""
import argparse,hashlib,json,shutil,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from tinygrad import Tensor,Device,Context
from tinygrad.nn.state import get_state_dict
from flare_q6a.runners.ppo import ActorCritic,load_policy
from flare_q6a.envs.surrogate_env import CableEnv,ROOT
from flare_q6a.utils.evaluation_metrics import EvaluationMetrics
from flare_q6a.utils.config import RIGID_TASK,FLEX_TASK,task_signature

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',default='records/ppo_genesis/model.npz')
    p.add_argument('--steps',type=int,help='maximum simulator control steps; default 100, or enough for requested episodes')
    p.add_argument('--episodes',type=int,help='total complete episodes, balanced across environments')
    p.add_argument('--physics',choices=['surrogate','genesis'],default='genesis')
    p.add_argument('--envs',type=int,default=2);p.add_argument('--test-episode-steps',type=int,default=0)
    p.add_argument('-e','--exp_name',choices=[FLEX_TASK,RIGID_TASK],default=RIGID_TASK)
    p.add_argument('-v','--vis',action='store_true');p.add_argument('--mode',choices=['mean','sample'],default='sample',help='sample adds learned Gaussian exploration noise (default); mean disables it')
    p.add_argument('--seed',type=int,default=123);p.add_argument('--output')
    a=p.parse_args()
    if a.envs<1 or (a.episodes is not None and a.episodes<1) or (a.steps is not None and a.steps<1):p.error('envs/episodes/steps must be positive')
    if a.seed<0:p.error('seed must be non-negative')
    if a.test_episode_steps<0:p.error('test-episode-steps must be non-negative')
    from flare_q6a.utils.hardware import require_adreno
    hardware=require_adreno();assert Device.DEFAULT=='CL'
    out=Path(a.output).resolve() if a.output else ROOT/'records'/datetime.now(timezone.utc).strftime('run_eval_'+a.exp_name+'_%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True,exist_ok=False)
    # The training writer replaces its checkpoint atomically. Copy an opened version
    # so evaluation always uses one fixed checkpoint even if training later advances.
    source=Path(a.checkpoint).resolve();snapshot=out/'checkpoint.npz';shutil.copyfile(source,snapshot)
    sha=hashlib.sha256(snapshot.read_bytes()).hexdigest();meta={}
    with np.load(snapshot,allow_pickle=False) as d:
        if 'metadata' in d:meta=json.loads(str(d['metadata']))
    if meta:
        run=meta.get('run',{})
        if run.get('experiment','s1_waypoint_passing')!=a.exp_name:raise ValueError('checkpoint experiment mismatch')
        if run.get('physics',a.physics)!=a.physics:raise ValueError('checkpoint physics mismatch')
        if a.exp_name==RIGID_TASK and run.get('task_signature')!=task_signature(True):raise ValueError('checkpoint task/model mismatch')
    manifest=dict(started_utc=datetime.now(timezone.utc).isoformat(),arguments=vars(a),checkpoint_source=str(source),
        checkpoint_sha256=sha,checkpoint_iteration=meta.get('iteration'),hardware=hardware,
        mode=a.mode,seed=a.seed,training_updates=False)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(f'Evaluation output: {out}\nFrozen checkpoint iteration: {meta.get("iteration")} | policy mode: {a.mode}',flush=True)
    np.random.seed(a.seed);Tensor.manual_seed(a.seed)
    model=ActorCritic(26);load_policy(model,snapshot)
    before={k:v.numpy().copy() for k,v in get_state_dict(model).items()}
    if a.exp_name==RIGID_TASK and a.physics!='genesis':raise ValueError('rigid waypoint task requires Genesis')
    if a.physics=='genesis':
        from flare_q6a.envs.genesis_bridge import GenesisBridge,torch
        torch.manual_seed(a.seed)
        env=GenesisBridge(a.envs,show_viewer=a.vis,rigid=a.exp_name==RIGID_TASK)
        if a.test_episode_steps:env.env.max_episode_length=a.test_episode_steps
        limit=env.env.max_episode_length+1;dt=env.env.dt
    else:env=CableEnv(a.envs);limit=1500;dt=.01
    budget=a.steps or (limit*((a.episodes+a.envs-1)//a.envs) if a.episodes else 100)
    metrics=EvaluationMetrics(a.envs,dt,a.episodes);obs=env.obs();started=time.perf_counter()
    with (out/'episodes.jsonl').open('w') as log,Context(TRAINING=0):
        for step in range(budget):
            mean,_=model.forward(obs)
            action=mean if a.mode=='mean' else mean+Tensor.randn(a.envs,4)*model.std
            action=action.detach().realize();raw_action=action.numpy()
            obs,r,d,to,terminal=env.step(action)
            assert np.isfinite(obs.numpy()).all() and np.isfinite(terminal.numpy()).all()
            reward=r.numpy();assert np.isfinite(reward).all() and np.isfinite(raw_action).all()
            rows=metrics.observe(reward,d.numpy(),to.numpy(),getattr(env,'last_step_info',{}),raw_action)
            for row in rows:
                log.write(json.dumps(row)+'\n');log.flush()
                print(f'Episode env={row["env_id"]} length={row["seconds"]:.2f}s return={row["reward"]:.3f} '+
                      f'timeout={row["timeout_only"]} causes={",".join(row["causes"]) or "none"}',flush=True)
            progress=dict(step=step+1,budget=budget,completed_episodes=len(metrics.rows),updated_unix=time.time())
            temp=out/'progress.json.tmp';temp.write_text(json.dumps(progress));temp.replace(out/'progress.json')
            if (step+1)%50==0:print(f'Evaluation step {step+1}/{budget}, completed episodes {len(metrics.rows)}',flush=True)
            if a.episodes and not metrics.active().any():break
    unchanged=all(np.array_equal(before[k],v.numpy()) for k,v in get_state_dict(model).items());assert unchanged
    result=dict(status='PASS',scope='frozen policy GPU evaluation; not a convergence certificate',
        experiment=a.exp_name,mode=a.mode,checkpoint_iteration=meta.get('iteration'),checkpoint_sha256=sha,
        seed=a.seed,steps=step+1,episode_horizon_seconds=limit*dt,test_episode_steps=a.test_episode_steps,seconds=time.perf_counter()-started,weights_unchanged=unchanged,**metrics.summary())
    if a.episodes and not result['requested_episodes_complete']:result['status']='INCOMPLETE'
    if a.test_episode_steps:assert result['safe_full_duration_episodes']>0,'timeout path not exercised'
    (out/'evaluation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)
    return 0 if result['status']=='PASS' else 2

if __name__=='__main__':raise SystemExit(main())
