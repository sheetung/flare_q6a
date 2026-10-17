"""OpenCL adaptation of rsl_rl v1.0.2 PPO. See LICENSE.rsl_rl.
Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES; ETH Zurich, Nikita Rudin.
"""
import json
import math
import os
import random
import sys
from pathlib import Path
import numpy as np
from tinygrad import Tensor, Context, nn
from tinygrad.nn.optim import Adam
from tinygrad.nn.state import get_parameters, get_state_dict

class ActorCritic:
    def __init__(self):
        self.actor=[nn.Linear(26,128),nn.Linear(128,128),nn.Linear(128,4)]
        self.critic=[nn.Linear(26,128),nn.Linear(128,128),nn.Linear(128,1)]
        # rsl_rl optimizes sigma directly, not log(sigma).
        self.std=Tensor.ones(4)
    def forward(self,obs):
        def run(ls):return ls[2](ls[1](ls[0](obs).tanh()).tanh())
        return run(self.actor),run(self.critic).squeeze(-1)
    def logprob(self,a,mean):
        return (-.5*((a-mean)/self.std).square()-self.std.log()-.5*math.log(2*math.pi)).sum(-1)

def adapt_lr(lr,kl,cfg):
    if cfg.get('schedule')=='adaptive' and cfg.get('desired_kl') is not None:
        if kl>cfg['desired_kl']*2: return max(1e-5,lr/1.5)
        if 0<kl<cfg['desired_kl']/2:return min(1e-2,lr*1.5)
    return lr

def losses(model,batch,cfg):
    mean,v=model.forward(batch['obs'])
    lp=model.logprob(batch['actions'],mean)
    ratio=(lp-batch['oldlog']).exp()
    surrogate=-(ratio*batch['advantages']).minimum(ratio.clip(1-cfg['clip_param'],1+cfg['clip_param'])*batch['advantages']).mean()
    error=(v-batch['returns']).square()
    if cfg['use_clipped_value_loss']:
        vc=batch['oldvalue']+(v-batch['oldvalue']).clip(-cfg['clip_param'],cfg['clip_param'])
        error=error.maximum((vc-batch['returns']).square())
    vl=error.mean()
    entropy=(model.std.log()+.5*math.log(2*math.pi*math.e)).sum()
    kl=((model.std/batch['oldsigma']+1e-5).log()
        +(batch['oldsigma'].square()+(batch['oldmean']-mean).square())/(2*model.std.square())-.5).sum(-1).mean()
    total=surrogate+cfg['value_loss_coef']*vl-cfg['entropy_coef']*entropy
    return total,surrogate,vl,entropy,kl

class PPO:
    def __init__(self,model,cfg,seed=1):
        self.model,self.cfg=model,dict(cfg)
        self.params=get_parameters(model)
        assert all(p.device.startswith('CL') for p in self.params)
        self.lr=cfg['learning_rate']
        self.opt=Adam(self.params,lr=self.lr,fused=False)
        self.rng=np.random.default_rng(seed)
        self.iteration,self.updates=0,0
    def update(self,batch,epochs=None,permutation=None):
        cfg=self.cfg;epochs=epochs or cfg['num_learning_epochs']
        count=batch['obs'].shape[0];mb=cfg['num_mini_batches'];size=count//mb
        if size<1:raise ValueError('rollout must contain at least num_mini_batches samples')
        # Match v1.0.2: one permutation per update, reused across epochs;
        # permutes only the truncated sample count if batch is not divisible.
        perm=self.rng.permutation(mb*size) if permutation is None else np.asarray(permutation)
        assert sorted(perm.tolist())==list(range(mb*size))
        stats=[]
        with Context(TRAINING=1):
            for _ in range(epochs):
                for i in range(mb):
                    idx=Tensor(perm[i*size:(i+1)*size].astype(np.int32),device='CL')
                    b={k:v[idx] for k,v in batch.items()}
                    self.opt.zero_grad()
                    total,policy,vl,entropy,kl=losses(self.model,b,cfg)
                    klval=float(kl.detach().item())
                    self.lr=adapt_lr(self.lr,klval,cfg)
                    self.opt.lr.assign([self.lr]).realize()
                    total.backward()
                    norm=(sum(p.grad.square().sum() for p in self.params)+1e-12).sqrt()
                    factor=(cfg['max_grad_norm']/(norm+1e-6)).clip(0,1).realize()
                    for p in self.params:p.grad=(p.grad*factor).realize()
                    self.opt.step();self.updates+=1
                    stats.append([float(t.detach().item()) for t in (total,policy,vl,entropy)]+[klval])
        values=np.asarray(stats);assert np.isfinite(values).all()
        assert np.isfinite(self.model.std.numpy()).all() and (self.model.std.numpy()>0).all(), 'non-positive policy sigma'
        self.iteration+=1
        return dict(zip(('loss','policy_loss','value_loss','entropy','kl'),values.mean(0).tolist()),
                    learning_rate=self.lr,updates=self.updates,discarded_samples=count-mb*size)
    def _optimizer_tensors(self):
        return {'lr':self.opt.lr,'b1_t':self.opt.b1_t,'b2_t':self.opt.b2_t,
                **{f'm.{i}':v for i,v in enumerate(self.opt.m)},**{f'v.{i}':v for i,v in enumerate(self.opt.v)}}
    def save(self,path,run_metadata=None):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        arrays={f'model.{k}':v.numpy() for k,v in get_state_dict(self.model).items()}
        arrays.update({f'optimizer.{k}':v.numpy() for k,v in self._optimizer_tensors().items()})
        for name,states in [('seed',Tensor._device_seeds),('counter',Tensor._device_rng_counters)]:
            arrays.update({f'rng.{name}.{k}':v.numpy() for k,v in states.items()})
        if 'torch' in sys.modules: arrays['rng.torch']=sys.modules['torch'].get_rng_state().numpy()
        meta={'schema':2,'iteration':self.iteration,'updates':self.updates,'learning_rate':self.lr,
              'config':self.cfg,'numpy_rng':self.rng.bit_generator.state,'tinygrad_seed':Tensor._seed,
              'python_rng':random.getstate(),'environment_restored':False,'run':run_metadata or {}}
        arrays['metadata']=np.array(json.dumps(meta))
        temp=path.with_suffix(path.suffix+'.tmp')
        with temp.open('wb') as f:np.savez(f,**arrays)
        os.replace(temp,path)
    def load(self,path):
        with np.load(path,allow_pickle=False) as data:
            meta=json.loads(str(data['metadata']))
            if meta['schema']!=2:raise ValueError('unsupported training checkpoint schema')
            if meta['config']!=self.cfg:raise ValueError('PPO config mismatch; refusing incompatible optimizer resume')
            tensors={f'model.{k}':v for k,v in get_state_dict(self.model).items()}
            tensors.update({f'optimizer.{k}':v for k,v in self._optimizer_tensors().items()})
            # Validate everything before mutating the model.
            for k,t in tensors.items():
                if k not in data or data[k].shape!=t.shape or not np.isfinite(data[k]).all():raise ValueError(f'invalid checkpoint tensor: {k}')
            for k,t in tensors.items():t.assign(Tensor(data[k],device=t.device)).realize()
            self.lr=meta['learning_rate'];self.iteration=meta['iteration'];self.updates=meta['updates']
            self.rng.bit_generator.state=meta['numpy_rng']
            def tuples(x):return tuple(tuples(v) for v in x) if isinstance(x,list) else x
            random.setstate(tuples(meta['python_rng']))
            Tensor.manual_seed(meta['tinygrad_seed'])
            for prefix,attr in [('rng.seed.', '_device_seeds'),('rng.counter.','_device_rng_counters')]:
                setattr(Tensor,attr,{k[len(prefix):]:Tensor(data[k],device=k[len(prefix):]) for k in data.files if k.startswith(prefix)})
            if 'rng.torch' in data and 'torch' in sys.modules:
                sys.modules['torch'].set_rng_state(sys.modules['torch'].from_numpy(data['rng.torch'].copy()))
        return meta

def load_policy(model,path):
    with np.load(path,allow_pickle=False) as data:
        state=get_state_dict(model)
        for k,t in state.items():
            key='model.'+k if 'metadata' in data else k
            if k=='std' and key not in data and 'log_std' in data:
                arr=np.exp(data['log_std']) # migrate old inference-only prototype
            else:arr=data[key]
            if arr.shape!=t.shape:raise ValueError(k)
            t.assign(Tensor(arr,device=t.device)).realize()
