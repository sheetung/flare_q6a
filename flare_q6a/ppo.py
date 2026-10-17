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
from tinygrad import Tensor, Context, nn, TinyJit
from tinygrad.nn.optim import Adam
from tinygrad.nn.state import get_parameters, get_state_dict
from hardware import require_adreno

class ActorCritic:
    def __init__(self,num_obs=26):
        def linear(inputs,outputs):
            layer=nn.Linear(inputs,outputs)
            # Consume each random initialization before constructing the next
            # layer; lazy equal-shaped random tensors can otherwise alias.
            layer.weight.realize();layer.bias.realize()
            return layer
        self.actor=[linear(num_obs,128),linear(128,128),linear(128,4)]
        self.critic=[linear(num_obs,128),linear(128,128),linear(128,1)]
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
    def __init__(self,model,cfg,seed=1,microbatch_size=100):
        if microbatch_size<1 or (microbatch_size>4096 and os.environ.get("RL_GPU_EXPERIMENTAL_CHUNKS")!="1"):
            raise ValueError("GPU microbatch size must be between 1 and 4096; larger kernels require internal experimental opt-in.")
        self.microbatch_size=microbatch_size
        self.inference_chunk_size=min(1024,max(128,microbatch_size))
        self.hardware=require_adreno()
        self.model,self.cfg=model,dict(cfg)
        self.params=get_parameters(model)
        assert all(p.device.startswith('CL') for p in self.params)
        # JIT inputs must own separate buffers, including freshly initialized
        # or checkpoint-loaded tensors with identical values/shapes.
        for p in self.params:p.assign(p.clone()).realize()
        self.batch_realize=os.environ.get('RL_PPO_BATCH_REALIZE','0')=='1'
        self.jit_enabled=os.environ.get('RL_PPO_JIT','0')=='1'
        self._chunk_runners={}
        self.rollout_jit=os.environ.get('RL_ROLLOUT_JIT','0')=='1'
        self._sample_runners={}
        self.jit_adam=os.environ.get('RL_PPO_JIT_ADAM','0')=='1'
        self._optimizer_runner=TinyJit(self._optimizer_compute) if self.jit_adam else self._optimizer_compute
        self.lr=cfg['learning_rate']
        self.opt=Adam(self.params,lr=self.lr,fused=False)
        for t in self._optimizer_tensors().values():t.assign(t.clone()).realize()
        self.rng=np.random.default_rng(seed)
        self.iteration,self.updates=0,0
    def _sample_compute(self,obs,noise,*parameters):
        mean,value=self.model.forward(obs)
        sigma=self.model.std.detach()+Tensor.zeros(obs.shape[0],4)
        action=(mean+noise*self.model.std).detach()
        logp=self.model.logprob(action,mean).detach()
        outputs=(action,logp,value.detach(),mean.detach(),sigma)
        Tensor.realize(*outputs)
        return outputs

    def sample(self,obs,noise):
        pieces=[]
        for start in range(0,obs.shape[0],self.inference_chunk_size):
            end=min(start+self.inference_chunk_size,obs.shape[0]);n=end-start
            x,z=(obs,noise) if start==0 and end==obs.shape[0] else (obs[start:end].clone().realize(),noise[start:end].clone().realize())
            if self.rollout_jit:
                if n not in self._sample_runners:self._sample_runners[n]=TinyJit(self._sample_compute)
                outputs=self._sample_runners[n](x,z,*self.params)
                # Trajectory storage must not alias reusable JIT outputs.
                outputs=tuple(t.clone() for t in outputs)
                Tensor.realize(*outputs)
            else:outputs=self._sample_compute(x,z,*self.params)
            pieces.append(outputs)
        if len(pieces)==1:return pieces[0]
        result=tuple(Tensor.cat(*items,dim=0) for items in zip(*pieces))
        Tensor.realize(*result)
        return result

    def value(self,obs):
        # Same bounded inference path, with no random-number consumption.
        return self.sample(obs,Tensor.zeros(obs.shape[0],4).realize())[2]

    def _chunk(self,batch,idx,n,size):
        keys=tuple(batch)
        def compute(index,*inputs):
            values=inputs[:len(keys)]
            self.opt.zero_grad()
            b={k:v[index] for k,v in zip(keys,values)}
            total,policy,vl,entropy,kl=losses(self.model,b,self.cfg)
            metrics=Tensor.stack(*[t.detach() for t in (total,policy,vl,entropy,kl)])
            (total*(n/size)).backward()
            grads=[p.grad.detach() for p in self.params]
            if self.batch_realize or self.jit_enabled:Tensor.realize(*grads,metrics)
            else:
                grads=[g.realize() for g in grads]
                metrics.realize()
            return tuple(grads),metrics
        if not self.jit_enabled:return compute(idx,*batch.values(),*self.params)
        key=(keys,n,size,tuple(v.shape for v in batch.values()))
        if key not in self._chunk_runners:self._chunk_runners[key]=TinyJit(compute)
        return self._chunk_runners[key](idx,*batch.values(),*self.params)

    def _optimizer_compute(self,*inputs):
        for p,g in zip(self.params,inputs[:len(self.params)]):p.grad=g
        norm=(sum(p.grad.square().sum() for p in self.params)+1e-12).sqrt()
        factor=(self.cfg['max_grad_norm']/(norm+1e-6)).clip(0,1).realize()
        if self.batch_realize:
            for p in self.params:p.grad=p.grad*factor
            Tensor.realize(*[p.grad for p in self.params])
        else:
            for p in self.params:p.grad=(p.grad*factor).realize()
        norm.realize()
        self.opt.step()
        return (norm,*self.params,*self._optimizer_tensors().values())

    def update(self,batch,epochs=None,permutation=None,progress_callback=None):
        cfg=self.cfg;epochs=epochs or cfg['num_learning_epochs']
        count=batch['obs'].shape[0];mb=cfg['num_mini_batches'];size=count//mb
        if size<1:raise ValueError('rollout must contain at least num_mini_batches samples')
        # Match v1.0.2: one permutation per update, reused across epochs;
        # permutes only the truncated sample count if batch is not divisible.
        perm=self.rng.permutation(mb*size) if permutation is None else np.asarray(permutation)
        assert sorted(perm.tolist())==list(range(mb*size))
        stats=[]
        with Context(TRAINING=1):
            for epoch in range(epochs):
                for i in range(mb):
                    # Split compute only: a logical minibatch still has one global
                    # gradient clip, one adaptive-KL decision and one Adam update.
                    accumulated=None
                    metric_tensor=None
                    for offset in range(0,size,self.microbatch_size):
                        n=min(self.microbatch_size,size-offset)
                        idx=Tensor(perm[i*size+offset:i*size+offset+n].astype(np.int32),device='CL')
                        if progress_callback:progress_callback('ppo_chunk',epoch=epoch+1,minibatch=i+1,
                            microbatch_offset=offset,microbatch_size=n,jit=self.jit_enabled)
                        grads,chunk_metrics=self._chunk(batch,idx,n,size)
                        weighted=chunk_metrics*(n/size)
                        metric_tensor=weighted if metric_tensor is None else metric_tensor+weighted
                        metric_tensor.realize()
                        # Returned JIT buffers are reused; accumulate before replay.
                        if accumulated is None:
                            # Own buffers: a future JIT replay overwrites its outputs.
                            accumulated=[g.clone().realize() for g in grads]
                        elif self.batch_realize:
                            accumulated=[a+g for a,g in zip(accumulated,grads)]
                            Tensor.realize(*accumulated)
                        else:accumulated=[(a+g).realize() for a,g in zip(accumulated,grads)]
                    for p,g in zip(self.params,accumulated):p.grad=g
                    # One host read per logical minibatch, rather than a queue
                    # synchronization for every compute chunk.
                    metrics=metric_tensor.numpy().astype(float)
                    if not np.isfinite(metrics).all():
                        raise FloatingPointError(f'PPO invalid minibatch losses: epoch={epoch+1}, minibatch={i+1}, values={metrics.tolist()}')
                    klval=float(metrics[4])
                    self.lr=adapt_lr(self.lr,klval,cfg)
                    self.opt.lr.assign([self.lr]).realize()
                    if progress_callback:progress_callback('ppo_optimizer',epoch=epoch+1,minibatch=i+1)
                    # Explicit inputs prevent JIT from retaining old parameter,
                    # learning-rate or optimizer-state buffers after assignment.
                    result=self._optimizer_runner(*accumulated,*self.params,*self._optimizer_tensors().values())
                    self.updates+=1
                    norm_value=float(result[0].item())
                    if not math.isfinite(norm_value):
                        raise FloatingPointError(f'PPO invalid gradient norm: epoch={epoch+1}, minibatch={i+1}, size={size}, gradient_norm={norm_value}')
                    sigma=self.model.std.numpy()
                    if progress_callback:progress_callback('ppo_checked',epoch=epoch+1,minibatch=i+1,
                        minibatch_size=size,microbatch_size=self.microbatch_size,sigma=sigma.tolist(),
                        gradient_norm=norm_value,learning_rate=self.lr)
                    if not np.isfinite(sigma).all() or not (sigma>0).all():
                        raise FloatingPointError(f'PPO invalid policy sigma after update {self.updates}: epoch={epoch+1}, minibatch={i+1}, size={size}, sigma={sigma.tolist()}, gradient_norm={norm_value}, lr={self.lr}')
                    stats.append(metrics.tolist())
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
            for k,t in tensors.items():t.assign(Tensor(data[k],device=t.device).clone()).realize()
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
