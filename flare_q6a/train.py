"""FLARE Genesis Vulkan physics + OpenCL PPO. No flight hardware access."""
import argparse,json,time
from pathlib import Path
import yaml
import numpy as np
from tinygrad import Tensor,Device,Context
from tinygrad.nn.state import get_state_dict
from env import CableEnv,ROOT
from ppo import ActorCritic,PPO

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--envs',type=int,default=16)
    p.add_argument('--steps',type=int,default=100)
    p.add_argument('--iterations',type=int,default=3,help='additional iterations, including on resume')
    p.add_argument('--epochs',type=int)
    p.add_argument('--minibatches',type=int)
    p.add_argument('--output')
    p.add_argument('--resume',help='schema-2 training checkpoint; environment starts fresh')
    p.add_argument('--physics',choices=['surrogate','genesis'],default='genesis')
    p.add_argument('--timeout-bootstrap',choices=['rsl','terminal'],default='rsl')
    args=p.parse_args()
    assert min(args.envs,args.steps,args.iterations)>0
    assert Device.DEFAULT=='CL','CPU training fallback forbidden'
    out=Path(args.output or ROOT/'records'/('run_'+str(time.time_ns()))).resolve();out.mkdir(parents=True,exist_ok=True)
    resume=Path(args.resume).resolve() if args.resume else None
    cfg=yaml.safe_load((ROOT/'Flare/config/train/s1_train_config.yaml').read_text())['algorithm']
    if args.epochs is not None:cfg['num_learning_epochs']=args.epochs
    if args.minibatches is not None:cfg['num_mini_batches']=args.minibatches
    cfg['timeout_bootstrap']=args.timeout_bootstrap
    assert cfg['num_learning_epochs']>0 and 0<cfg['num_mini_batches']<=args.envs*args.steps
    # Validate checkpoint mode before expensive scene construction.
    if resume:
        with np.load(resume,allow_pickle=False) as d:meta=json.loads(str(d['metadata']))
        if meta['run']['physics']!=args.physics:raise ValueError('resume physics mismatch')
    Tensor.manual_seed(1);setup_start=time.perf_counter()
    if args.physics=='genesis':
        from genesis_bridge import GenesisBridge
        env=GenesisBridge(args.envs)
    else:env=CableEnv(args.envs)
    model=ActorCritic();trainer=PPO(model,cfg)
    if resume:trainer.load(resume)
    obs=env.obs();started=time.perf_counter();history=[]
    first=trainer.params[0].numpy().copy()
    setup_seconds=started-setup_start
    for _ in range(args.iterations):
        ts=time.perf_counter();roll=[]
        with Context(TRAINING=0):
            for _ in range(args.steps):
                mean,value=model.forward(obs)
                oldmean=mean.detach().realize()
                oldsigma=(model.std.detach()+Tensor.zeros(args.envs,4)).realize()
                action=(mean+Tensor.randn(args.envs,4)*model.std).detach().realize()
                logp=model.logprob(action,mean).detach().realize();value=value.detach().realize()
                newobs,reward,done,timeout,terminal_obs=env.step(action)
                if args.timeout_bootstrap=='terminal':
                    _,boot=model.forward(terminal_obs);boot=boot.detach()
                else:boot=value # exact rsl_rl v1.0.2 process_env_step convention
                reward=(reward+cfg['gamma']*timeout.float()*boot).realize()
                roll.append((obs,action,logp,value,reward,done,oldmean,oldsigma));obs=newobs
            _,nv=model.forward(obs);nv=nv.detach().realize();adv=Tensor.zeros(args.envs)
            advantages=[];returns=[]
            for item in reversed(roll):
                v,r,d=item[3:6]
                adv=(r+cfg['gamma']*(~d).float()*nv-v+cfg['gamma']*cfg['lam']*(~d).float()*adv).detach().realize()
                advantages.append(adv);returns.append((adv+v).realize());nv=v
        def flatten(index,width=None):
            return Tensor.stack(*[t[index] for t in roll]).reshape(-1,*(() if width is None else (width,))).detach().realize()
        ad=Tensor.stack(*advantages[::-1]).reshape(-1).realize()
        batch={'obs':flatten(0,26),'actions':flatten(1,4),'oldlog':flatten(2),'oldvalue':flatten(3),
               'oldmean':flatten(6,4),'oldsigma':flatten(7,4),
               'advantages':((ad-ad.mean())/(ad.std(correction=1)+1e-8)).detach().realize(),
               'returns':Tensor.stack(*returns[::-1]).reshape(-1).detach().realize()}
        stats=trainer.update(batch)
        assert np.isfinite(obs.numpy()).all()
        row=dict(iteration=trainer.iteration,samples=args.envs*args.steps,seconds=time.perf_counter()-ts,
                 resets=int(flatten(5).float().sum().item()),device='CL',**stats)
        history.append(row);print(json.dumps(row),flush=True)
        run={'physics':args.physics,'envs':args.envs,'steps':args.steps,'environment_resume':'fresh reset'}
        trainer.save(out/'checkpoint.npz',run)
        (out/'metrics.json').write_text(json.dumps(history,indent=2))
    np.savez(out/'model.npz',**{k:v.numpy() for k,v in get_state_dict(model).items()})
    delta=float(np.abs(trainer.params[0].numpy()-first).max());assert delta>0
    result=dict(status='PASS',scope=args.physics+' PPO training smoke; convergence unverified',max_weight_change=delta,
                setup_seconds=setup_seconds,training_seconds=time.perf_counter()-started,arguments=vars(args),config=cfg,
                iteration=trainer.iteration,updates=trainer.updates)
    (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
