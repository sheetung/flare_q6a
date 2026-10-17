"""Frozen rollout outputs and changed weights against eager OpenCL inference."""
import os,json
import numpy as np
from tinygrad import Tensor,Context
from ppo import ActorCritic,PPO
from task_config import train_config
Tensor.manual_seed(17)
model=ActorCritic();cfg=train_config()['algorithm']
os.environ['RL_ROLLOUT_JIT']='1'
trainer=PPO(model,cfg)
rng=np.random.default_rng(19)
max_error=0.
with Context(TRAINING=0):
    saved=None
    for step in range(12):
        obs=Tensor(rng.normal(size=(64,26)).astype(np.float32));noise=Tensor(rng.normal(size=(64,4)).astype(np.float32))
        if step in (4,8):
            for p in trainer.params:p.assign(p+0.001).realize()
        actual=trainer.sample(obs,noise)
        expected=trainer._sample_compute(obs,noise,*trainer.params)
        for a,b in zip(actual,expected):
            av,bv=a.numpy(),b.numpy();max_error=max(max_error,float(np.abs(av-bv).max()))
            np.testing.assert_allclose(av,bv,rtol=2e-5,atol=2e-5)
        if saved is None:saved=actual;snapshot=[t.numpy().copy() for t in actual]
        else:
            for t,a in zip(saved,snapshot):np.testing.assert_array_equal(t.numpy(),a)
print(json.dumps({'status':'PASS','device':'CL','envs':64,'steps':12,'max_absolute_error':max_error,
    'checks':['fresh observations and Gaussian noise','explicit updated weights','trajectory buffers preserved across replay']}))
