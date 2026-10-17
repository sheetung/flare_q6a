"""Frozen rollout outputs and changed weights against eager OpenCL inference."""
import os,json,math,argparse
import torch
torch.set_num_threads(2)
import numpy as np
from tinygrad import Tensor,Context
from ppo import ActorCritic,PPO
from task_config import train_config
ap=argparse.ArgumentParser();ap.add_argument('--envs',type=int,default=64);args=ap.parse_args()
Tensor.manual_seed(17)
model=ActorCritic();cfg=train_config()['algorithm']
os.environ['RL_ROLLOUT_JIT']='1'
trainer=PPO(model,cfg)
trainer.params[0].numpy() # same early weight snapshot as train.py
assert not np.array_equal(model.actor[1].weight.numpy(),model.critic[1].weight.numpy()), "random layer initializations alias"
inputs=[*trainer.params,*trainer._optimizer_tensors().values()]
assert len({t.realize().uop.base for t in inputs})==len(inputs), "model/optimizer buffers alias"
rng=np.random.default_rng(19)
max_error=0.
with Context(TRAINING=0):
    saved=None
    for step in range(12):
        obs=Tensor(rng.normal(size=(args.envs,26)).astype(np.float32));noise=Tensor(rng.normal(size=(args.envs,4)).astype(np.float32))
        if step in (4,8):
            for p in trainer.params:p.assign(p+0.001).realize()
        actual=trainer.sample(obs,noise)
        def forward(layers):
            x=torch.from_numpy(obs.numpy().copy())
            for i,l in enumerate(layers):
                x=x@torch.from_numpy(l.weight.numpy().copy()).T+torch.from_numpy(l.bias.numpy().copy())
                if i<2:x=x.tanh()
            return x
        mean=forward(model.actor);value=forward(model.critic).squeeze(-1)
        sigma=torch.from_numpy(model.std.numpy().copy()).expand(args.envs,4)
        action=mean+torch.from_numpy(noise.numpy().copy())*sigma
        logp=(-.5*((action-mean)/sigma).square()-sigma.log()-.5*math.log(2*math.pi)).sum(-1)
        expected=tuple(t.numpy() for t in (action,logp,value,mean,sigma))
        for a,b in zip(actual,expected):
            av,bv=a.numpy(),b;max_error=max(max_error,float(np.abs(av-bv).max()))
            np.testing.assert_allclose(av,bv,rtol=2e-5,atol=2e-5)
        if saved is None:saved=actual;snapshot=[t.numpy().copy() for t in actual]
        else:
            for t,a in zip(saved,snapshot):np.testing.assert_array_equal(t.numpy(),a)
print(json.dumps({'status':'PASS','device':'CL','envs':args.envs,'steps':12,'max_absolute_error':max_error,
    'checks':['independent layer random initialization','independent fresh model and optimizer buffers','independent CPU reference for GPU rollout','bounded inference chunks and tail','fresh observations and Gaussian noise','explicit updated weights','trajectory buffers preserved across replay']}))
