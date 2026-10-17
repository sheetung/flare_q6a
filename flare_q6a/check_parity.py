"""Actual rsl_rl v1.0.2 CPU reference vs OpenCL PPO; CPU is test-only."""
import sys,json,tempfile,math
from pathlib import Path
import numpy as np
import torch
from tinygrad import Tensor,Device,Context
from tinygrad.nn.state import get_state_dict
from env import ROOT
from ppo import ActorCritic,PPO,losses,adapt_lr
sys.path.insert(0,str(ROOT/'rsl_rl_reference'))
from rsl_rl.modules import ActorCritic as ReferenceActorCritic
from rsl_rl.algorithms import PPO as ReferencePPO
assert Device.DEFAULT=='CL';torch.set_num_threads(2);Tensor.manual_seed(7)
cfg=dict(clip_param=.2,desired_kl=.01,entropy_coef=.002,gamma=.99,lam=.95,learning_rate=.0003,
         max_grad_norm=1.,num_learning_epochs=2,num_mini_batches=2,schedule='adaptive',use_clipped_value_loss=True,value_loss_coef=1.)
m=ActorCritic();t=PPO(m,cfg)
r=ReferenceActorCritic(26,26,4,actor_hidden_dims=[128,128],critic_hidden_dims=[128,128],activation='tanh')
with torch.no_grad():
    for source,target in [(m.actor,r.actor),(m.critic,r.critic)]:
        for i,layer in enumerate(source):
            target[2*i].weight.copy_(torch.from_numpy(layer.weight.numpy().copy()))
            target[2*i].bias.copy_(torch.from_numpy(layer.bias.numpy().copy()))
    r.std.copy_(torch.from_numpy(m.std.numpy().copy()))
ref=ReferencePPO(r,**cfg);ref.init_storage(2,4,[26],[None],[4])
rng=np.random.default_rng(4)
obs=rng.normal(size=(8,26)).astype(np.float32)
with torch.no_grad():
    r.act(torch.from_numpy(obs));oldmean=r.action_mean.clone();oldsigma=r.action_std.clone()
    actions=oldmean+torch.from_numpy(rng.normal(size=(8,4)).astype(np.float32))*.8
    oldlog=r.get_actions_log_prob(actions)
    ov=r.evaluate(torch.from_numpy(obs)).squeeze(-1)
advantages=torch.from_numpy(rng.normal(size=8).astype(np.float32));returns=ov+torch.from_numpy(rng.normal(size=8).astype(np.float32))*.3
arrays=dict(obs=obs,actions=actions.numpy(),oldlog=oldlog.numpy(),oldvalue=ov.numpy(),oldmean=oldmean.numpy(),oldsigma=oldsigma.numpy(),advantages=advantages.numpy(),returns=returns.numpy())
batch={k:Tensor(v,device='CL') for k,v in arrays.items()}
# Loss + full, unclipped gradients against independent PyTorch expression.
with Context(TRAINING=1):
    total,*_=losses(m,batch,cfg);total.backward()
r.act(torch.from_numpy(obs));lp=r.get_actions_log_prob(actions);v=r.evaluate(torch.from_numpy(obs)).squeeze(-1)
ratio=(lp-oldlog).exp();policy=torch.maximum(-advantages*ratio,-advantages*ratio.clamp(.8,1.2)).mean()
vc=ov+(v-ov).clamp(-.2,.2);vl=torch.maximum((v-returns)**2,(vc-returns)**2).mean()
rtotal=policy+vl-.002*r.entropy.mean();ref.optimizer.zero_grad();rtotal.backward()
np.testing.assert_allclose(total.item(),rtotal.item(),rtol=2e-4,atol=2e-5)
gradmax=0.
for source,target in [(m.actor,r.actor),(m.critic,r.critic)]:
    for i,l in enumerate(source):
        for a,b in [(l.weight,target[2*i].weight),(l.bias,target[2*i].bias)]:
            gradmax=max(gradmax,float(np.abs(a.grad.numpy()-b.grad.numpy()).max()))
            np.testing.assert_allclose(a.grad.numpy(),b.grad.numpy(),rtol=3e-3,atol=3e-5)
np.testing.assert_allclose(m.std.grad.numpy(),r.std.grad.numpy(),rtol=3e-3,atol=3e-5)
# Populate actual upstream storage and give both implementations the same shuffle.
store=ref.storage
for name,key,width in [('observations','obs',26),('actions','actions',4),('values','oldvalue',1),('returns','returns',1),('actions_log_prob','oldlog',1),('advantages','advantages',1),('mu','oldmean',4),('sigma','oldsigma',4)]:
    getattr(store,name).copy_(torch.from_numpy(arrays[key]).reshape(4,2,width))
perm=np.array([3,6,0,5,2,1,7,4],np.int64)
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
            np.testing.assert_allclose(a.numpy(),b.detach().numpy(),rtol=5e-4,atol=5e-6)
np.testing.assert_allclose(m.std.numpy(),r.std.detach().numpy(),atol=5e-6)
assert abs(t.lr-ref.learning_rate)<1e-12
assert math.isclose(adapt_lr(.0003,.03,cfg),.0002)
assert math.isclose(adapt_lr(.0003,.001,cfg),.00045)
# Save/resume at an update boundary, then repeat same next update and RNG draws.
with tempfile.TemporaryDirectory() as folder:
    path=Path(folder)/'checkpoint.npz';t.save(path,{'physics':'test'})
    expected_noise=Tensor.randn(3,4).numpy();expected_permutation=t.rng.permutation(8)
    t.update(batch,permutation=perm)
    expected={k:v.numpy().copy() for k,v in get_state_dict(m).items()}
    restored=PPO(ActorCritic(),cfg);restored.load(path)
    np.testing.assert_array_equal(Tensor.randn(3,4).numpy(),expected_noise)
    np.testing.assert_array_equal(restored.rng.permutation(8),expected_permutation)
    restored.update(batch,permutation=perm)
    for k,v in get_state_dict(restored.model).items():np.testing.assert_allclose(v.numpy(),expected[k],atol=1e-7,rtol=1e-6)
    assert restored.updates==t.updates and restored.iteration==t.iteration
print(json.dumps({'status':'PASS','reference':'rsl_rl v1.0.2','gradient_max_absolute_error':gradmax,
                  'parameter_max_absolute_error':parammax,'learning_rate':t.lr,'checks':['loss','gradients','actual upstream PPO minibatch updates','adaptive KL','Adam resume','GPU RNG resume','shuffle RNG resume']}))
