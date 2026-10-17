"""Reload and evaluate only in the experimental GPU simulator."""
import argparse,json
import numpy as np
from tinygrad import Tensor, Device, Context
from tinygrad.nn.state import get_state_dict
from train import ActorCritic
from env import CableEnv

p=argparse.ArgumentParser();p.add_argument('--checkpoint',default='records/ppo_genesis/model.npz');p.add_argument('--steps',type=int,default=100)
p.add_argument('--physics',choices=['surrogate','genesis'],default='genesis')
p.add_argument('--envs',type=int,default=2)
p.add_argument('--test-episode-steps',type=int,default=0)
p.add_argument('-e','--exp_name',choices=['s1_waypoint_passing','rigid_hover'],default='s1_waypoint_passing')
p.add_argument('-v','--vis',action='store_true')
a=p.parse_args();assert Device.DEFAULT=='CL'
from hardware import require_adreno
require_adreno()
from ppo import load_policy
model=ActorCritic(19 if a.exp_name=='rigid_hover' else 26);load_policy(model,a.checkpoint)
if a.exp_name=='rigid_hover':
    if a.physics!='genesis':raise ValueError('rigid_hover requires Genesis')
    from rigid_hover import RigidHover
    env=RigidHover(a.envs,show_viewer=a.vis)
    if a.test_episode_steps:env.max_episode_length=a.test_episode_steps
elif a.physics=='genesis':
    from genesis_bridge import GenesisBridge
    env=GenesisBridge(a.envs,show_viewer=a.vis)
    if a.test_episode_steps: env.env.max_episode_length=a.test_episode_steps
else: env=CableEnv(a.envs)
obs=env.obs();rewards=[];resets=0;timeouts=0
with Context(TRAINING=0):
    for _ in range(a.steps):
        mean,_=model.forward(obs)
        obs,r,d,to,terminal=env.step(mean.detach().realize())
        timeouts+=int(to.float().sum().item())
        assert np.isfinite(terminal.numpy()).all()
        rewards.append(r)
        resets+=int(d.float().sum().item())
assert np.isfinite(obs.numpy()).all()
if a.test_episode_steps: assert timeouts>0, 'timeout path not exercised'
print(json.dumps({'status':'PASS','scope':'checkpoint reload and '+a.physics+' GPU evaluation',
                  'steps':a.steps,'envs':a.envs,'timeouts':timeouts,'device':Device.DEFAULT,'resets':resets,
                  'mean_reward':float(Tensor.stack(*rewards).mean().item())}))
