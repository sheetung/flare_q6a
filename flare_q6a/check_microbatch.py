"""Actual rsl_rl v1.0.2 CPU reference vs OpenCL PPO; CPU is test-only."""
import sys,json,math,argparse,time,os
import numpy as np
import torch
from tinygrad import Tensor,Device
from env import ROOT
from ppo import ActorCritic,PPO,adapt_lr
sys.path.insert(0,str(ROOT/'rsl_rl_reference'))
from rsl_rl.modules import ActorCritic as ReferenceActorCritic
from rsl_rl.algorithms import PPO as ReferencePPO
ap=argparse.ArgumentParser();ap.add_argument('--samples',type=int,default=1600);ap.add_argument('--microbatch-size',type=int,default=100);ap.add_argument('--epochs',type=int,default=2)
args=ap.parse_args();assert args.samples%4==0
assert Device.DEFAULT=='CL';torch.set_num_threads(2);Tensor.manual_seed(7)
started=time.perf_counter()
cfg=dict(clip_param=.2,desired_kl=.01,entropy_coef=.002,gamma=.99,lam=.95,learning_rate=.0003,
         max_grad_norm=1.,num_learning_epochs=args.epochs,num_mini_batches=4,schedule='adaptive',use_clipped_value_loss=True,value_loss_coef=1.)
m=ActorCritic();t=PPO(m,cfg,microbatch_size=args.microbatch_size)
r=ReferenceActorCritic(26,26,4,actor_hidden_dims=[128,128],critic_hidden_dims=[128,128],activation='tanh')
with torch.no_grad():
    for source,target in [(m.actor,r.actor),(m.critic,r.critic)]:
        for i,layer in enumerate(source):
            target[2*i].weight.copy_(torch.from_numpy(layer.weight.numpy().copy()))
            target[2*i].bias.copy_(torch.from_numpy(layer.bias.numpy().copy()))
    r.std.copy_(torch.from_numpy(m.std.numpy().copy()))
ref=ReferencePPO(r,**cfg);ref.init_storage(4,args.samples//4,[26],[None],[4])
rng=np.random.default_rng(4)
obs=rng.normal(size=(args.samples,26)).astype(np.float32)
with torch.no_grad():
    r.act(torch.from_numpy(obs));oldmean=r.action_mean.clone();oldsigma=r.action_std.clone()
    actions=oldmean+torch.from_numpy(rng.normal(size=(args.samples,4)).astype(np.float32))*.8
    oldlog=r.get_actions_log_prob(actions)
    ov=r.evaluate(torch.from_numpy(obs)).squeeze(-1)
advantages=torch.from_numpy(rng.normal(size=args.samples).astype(np.float32));returns=ov+torch.from_numpy(rng.normal(size=args.samples).astype(np.float32))*.3
arrays=dict(obs=obs,actions=actions.numpy(),oldlog=oldlog.numpy(),oldvalue=ov.numpy(),oldmean=oldmean.numpy(),oldsigma=oldsigma.numpy(),advantages=advantages.numpy(),returns=returns.numpy())
batch={k:Tensor(v,device='CL') for k,v in arrays.items()}
# Populate actual upstream storage and give both implementations the same shuffle.
store=ref.storage
for name,key,width in [('observations','obs',26),('actions','actions',4),('values','oldvalue',1),('returns','returns',1),('actions_log_prob','oldlog',1),('advantages','advantages',1),('mu','oldmean',4),('sigma','oldsigma',4)]:
    getattr(store,name).copy_(torch.from_numpy(arrays[key]).reshape(args.samples//4,4,width))
perm=rng.permutation(args.samples)
original_randperm=torch.randperm
torch.randperm=lambda *a,**kw:torch.tensor(perm,device=kw.get('device','cpu'))
try:ref.update()
finally:torch.randperm=original_randperm
stats=t.update(batch,permutation=perm)
parammax=0.
for source,target in [(m.actor,r.actor),(m.critic,r.critic)]:
    for i,l in enumerate(source):
        for a,b in [(l.weight,target[2*i].weight),(l.bias,target[2*i].bias)]:
            parammax=max(parammax,float(np.abs(a.numpy()-b.detach().numpy()).max()))
            np.testing.assert_allclose(a.numpy(),b.detach().numpy(),rtol=2e-3,atol=2e-5)
np.testing.assert_allclose(m.std.numpy(),r.std.detach().numpy(),atol=2e-5)
assert abs(t.lr-ref.learning_rate)<1e-12
assert math.isclose(adapt_lr(.0003,.03,cfg),.0002)
assert math.isclose(adapt_lr(.0003,.001,cfg),.00045)
# Compare Adam moments too, so clipping/weighting errors cannot hide behind parameters.
for k,p in enumerate(t.params):
    if p is m.std:rp=r.std
    else:
        rp=None
        for source,target in [(m.actor,r.actor),(m.critic,r.critic)]:
            for i,l in enumerate(source):
                if p is l.weight:rp=target[2*i].weight
                if p is l.bias:rp=target[2*i].bias
        assert rp is not None
    state=ref.optimizer.state[rp]
    np.testing.assert_allclose(t.opt.m[k].numpy(),state['exp_avg'].numpy(),rtol=3e-3,atol=2e-6)
    np.testing.assert_allclose(t.opt.v[k].numpy(),state['exp_avg_sq'].numpy(),rtol=3e-3,atol=2e-7)
assert t.updates==args.epochs*4
print(json.dumps({'status':'PASS','reference':'actual rsl_rl v1.0.2 CPU test-only',
    'training_device':'CL','jit':t.jit_enabled,'jit_adam':t.jit_adam,'batch_realize':t.batch_realize,'samples':args.samples,'logical_minibatch_size':args.samples//4,
    'microbatch_size':args.microbatch_size,'epochs':args.epochs,'adam_updates':t.updates,
    'parameter_max_absolute_error':parammax,'seconds':time.perf_counter()-started,
    'checks':['weighted gradient accumulation','global gradient clipping','Adam moments','adaptive KL','unchanged Adam update count']}))
