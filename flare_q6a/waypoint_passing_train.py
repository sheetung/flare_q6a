import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

"""FLARE Genesis Vulkan physics + OpenCL PPO. No flight hardware access."""
import argparse,json,time,faulthandler,signal
from pathlib import Path
from datetime import datetime,timezone
import yaml
import numpy as np
from tinygrad import Tensor,Device,Context
from tinygrad.nn.state import get_state_dict
from flare_q6a.envs.surrogate_env import CableEnv,ROOT
from flare_q6a.runners.ppo import ActorCritic,PPO
from flare_q6a.utils.hardware import configure_main_cpu
from flare_q6a.utils.metrics import ExperimentLogger
from flare_q6a.utils.config import RIGID_TASK,FLEX_TASK,SERVER_S2_TASK,DEFAULT_ENVS,DEFAULT_ITERATIONS,default_envs,train_config,task_signature

def main():
    faulthandler.enable()
    faulthandler.register(signal.SIGUSR1,all_threads=True)
    p=argparse.ArgumentParser()
    p.add_argument('-B','--num_envs','--envs',dest='envs',type=int,help=f'default {default_envs(FLEX_TASK)} for original flexible task; {DEFAULT_ENVS} for rigid')
    p.add_argument('--steps',type=int,help='default: upstream runner.num_steps_per_env (100)')
    p.add_argument('--max_iterations','--iterations',dest='iterations',type=int,default=DEFAULT_ITERATIONS,help='additional iterations, including on resume')
    p.add_argument('--epochs',type=int)
    p.add_argument('--minibatches',type=int)
    p.add_argument('--microbatch-size',type=int,help='GPU compute chunk; auto: min(logical minibatch, 4096), PPO update count unchanged')
    p.add_argument('--output',help='optional run directory; default: records/run_<UTC timestamp>')
    p.add_argument('--resume',help='schema-2 training checkpoint; environment starts fresh')
    p.add_argument('--physics',choices=['surrogate','genesis'],default='genesis')
    p.add_argument('--timeout-bootstrap',choices=['rsl','terminal'],default='rsl')
    p.add_argument('-e','--exp_name',choices=[FLEX_TASK,RIGID_TASK,SERVER_S2_TASK],default=FLEX_TASK)
    p.add_argument('-v','--vis',action='store_true',help='show original Genesis viewer and target')
    args=p.parse_args()
    if args.envs is None:args.envs=default_envs(args.exp_name)
    if args.exp_name in (RIGID_TASK,SERVER_S2_TASK) and args.physics!='genesis':p.error('rigid waypoint task requires Genesis Vulkan physics')
    source_train=train_config(args.exp_name)
    if args.steps is None:args.steps=source_train['runner']['num_steps_per_env']
    if args.vis and args.physics!='genesis':p.error('-v requires --physics genesis')
    assert min(args.envs,args.steps,args.iterations)>0
    assert Device.DEFAULT=='CL','CPU training fallback forbidden'
    out=Path(args.output or ROOT/'records'/datetime.now(timezone.utc).strftime('run_'+args.exp_name+'_%Y%m%d_%H%M%S_%f')).resolve();out.mkdir(parents=True,exist_ok=True)
    print(f'Training output: {out}',flush=True)
    def progress(phase,**extra):
        data={'phase':phase,'updated_unix':time.time(),**extra}
        temp=out/'progress.json.tmp';temp.write_text(json.dumps(data));temp.replace(out/'progress.json')
    progress('initializing')
    resume=Path(args.resume).resolve() if args.resume else None
    cfg=source_train['algorithm']
    if args.epochs is not None:cfg['num_learning_epochs']=args.epochs
    if args.minibatches is not None:cfg['num_mini_batches']=args.minibatches
    cfg['timeout_bootstrap']=args.timeout_bootstrap
    assert cfg['num_learning_epochs']>0 and 0<cfg['num_mini_batches']<=args.envs*args.steps
    logical_size=args.envs*args.steps//cfg['num_mini_batches']
    compute_chunk=args.microbatch_size if args.microbatch_size is not None else min(4096,logical_size)
    if not 1<=compute_chunk<=4096:p.error('--microbatch-size must be between 1 and 4096')
    # Validate checkpoint mode before expensive scene construction.
    if resume:
        with np.load(resume,allow_pickle=False) as d:meta=json.loads(str(d['metadata']))
        if meta['run']['physics']!=args.physics:raise ValueError('resume physics mismatch')
        if meta['run'].get('experiment','s1_waypoint_passing')!=args.exp_name:raise ValueError('resume experiment mismatch')
        if args.exp_name==RIGID_TASK and meta['run'].get('task_signature')!=task_signature(True):raise ValueError('resume rigid waypoint task/model mismatch')
        if args.exp_name==SERVER_S2_TASK:
            from flare_q6a.envs.server_s2_bridge import server_s2_signature
            if meta['run'].get('task_signature')!=server_s2_signature():raise ValueError('resume S2 task/model mismatch')
    Tensor.manual_seed(source_train['seed']);setup_start=time.perf_counter()
    if args.physics=='genesis':
        from flare_q6a.envs.genesis_bridge import GenesisBridge,torch
        torch.manual_seed(source_train['seed'])
        if args.exp_name==SERVER_S2_TASK:
            from flare_q6a.envs.server_s2_bridge import setup_server_s2
            setup_server_s2()
        env=GenesisBridge(args.envs,show_viewer=args.vis,rigid=args.exp_name==RIGID_TASK)
        # Original FLARE calls learn(init_at_random_ep_len=True).
        env.env.episode_length_buf=torch.randint_like(env.env.episode_length_buf,high=env.env.max_episode_length)
    else:env=CableEnv(args.envs)
    num_obs=env.obs().shape[1]
    policy=source_train['policy']
    model=ActorCritic(num_obs,policy['actor_hidden_dims'],policy['critic_hidden_dims'],policy['activation'],policy['init_noise_std']);trainer=PPO(model,cfg,microbatch_size=compute_chunk)
    cpu_staging_threads=torch.get_num_threads() if args.physics=='genesis' else None
    host_dispatch=configure_main_cpu()
    print(f'Host dispatch: {host_dispatch}',flush=True)
    snapshot={'arguments':vars(args),'algorithm':cfg,'num_obs':num_obs,'upstream_train_config':source_train,
              'execution':{'microbatch_size':trainer.microbatch_size,'inference_chunk_size':trainer.inference_chunk_size,'row_lookup':'native_opencl_index_load','cpu_staging_threads':cpu_staging_threads,'jit':trainer.jit_enabled,'jit_adam':trainer.jit_adam,'rollout_jit':trainer.rollout_jit,'batch_realize':trainer.batch_realize,'hardware':trainer.hardware,'host_dispatch':host_dispatch}}
    if args.physics=='genesis':snapshot.update(environment=env.source_config,task_signature=env.signature,random_initial_episode_ages=True)
    (out/'config.json').write_text(json.dumps(snapshot,indent=2))
    print(f'GPU PPO: logical minibatch={args.envs*args.steps//cfg["num_mini_batches"]}, compute chunk={trainer.microbatch_size}, '
          f'inference chunk={trainer.inference_chunk_size}, gradient JIT={trainer.jit_enabled}, Adam JIT={trainer.jit_adam}, rollout JIT={trainer.rollout_jit}, '
          f'Adam updates/iteration={cfg["num_learning_epochs"]*cfg["num_mini_batches"]}',flush=True)
    loaded_meta=trainer.load(resume) if resume else {}
    obs=env.obs();started=time.perf_counter();history=[]
    first=trainer.params[0].numpy().copy()
    setup_seconds=started-setup_start
    logger_state=loaded_meta.get('run',{}).get('logger',{})
    initial_samples=logger_state.get('total_samples',trainer.iteration*args.envs*args.steps)
    initial_time=logger_state.get('total_time',0.)
    if resume and not logger_state:
        try:
            old_rows=json.loads((resume.parent/'metrics.json').read_text())
            initial_time=sum(r['seconds'] for r in old_rows if r['iteration']<=trainer.iteration)
        except (OSError,ValueError,KeyError):pass
    logger=ExperimentLogger(out,args.envs,getattr(env,'dt',.01),trainer.iteration+args.iterations,initial_samples,initial_time)
    logger.restore_episodes(logger_state)
    print('Logging: original FLARE waypoint task; completed-episode window=100; waypoint counts are not episode success rates.',flush=True)
    save_interval=source_train['runner']['save_interval']
    target_iteration=trainer.iteration+args.iterations
    for _ in range(args.iterations):
        ts=time.perf_counter();roll=[];logger.begin_iteration()
        progress('rollout',iteration=trainer.iteration+1,step=0,steps=args.steps)
        with Context(TRAINING=0):
            for step in range(args.steps):
                action,logp,value,oldmean,oldsigma=trainer.sample(obs,Tensor.randn(args.envs,4).realize())
                newobs,reward,done,timeout,terminal_obs=env.step(action)
                logger.observe(reward,done,getattr(env,'last_step_info',None))
                if step % 10 == 0:progress('rollout',iteration=trainer.iteration+1,step=step+1,steps=args.steps)
                if args.timeout_bootstrap=='terminal':
                    boot=trainer.value(terminal_obs).detach()
                else:boot=value # exact rsl_rl v1.0.2 process_env_step convention
                reward=(reward+cfg['gamma']*timeout.float()*boot).realize()
                roll.append((obs,action,logp,value,reward,done,oldmean,oldsigma));obs=newobs
            collection_end=time.perf_counter()
            nv=trainer.value(obs).detach().realize();adv=Tensor.zeros(args.envs)
            advantages=[];returns=[]
            for item in reversed(roll):
                v,r,d=item[3:6]
                adv=(r+cfg['gamma']*(~d).float()*nv-v+cfg['gamma']*cfg['lam']*(~d).float()*adv).detach().realize()
                advantages.append(adv);returns.append((adv+v).realize());nv=v
        def flatten(index,width=None):
            return Tensor.stack(*[t[index] for t in roll]).reshape(-1,*(() if width is None else (width,))).detach().realize()
        ad=Tensor.stack(*advantages[::-1]).reshape(-1).realize()
        batch={'obs':flatten(0,num_obs),'actions':flatten(1,4),'oldlog':flatten(2),'oldvalue':flatten(3),
               'oldmean':flatten(6,4),'oldsigma':flatten(7,4),
               'advantages':((ad-ad.mean())/(ad.std(correction=1)+1e-8)).detach().realize(),
               'returns':Tensor.stack(*returns[::-1]).reshape(-1).detach().realize()}
        progress('ppo_update',iteration=trainer.iteration+1)
        try:
            stats=trainer.update(batch,progress_callback=lambda phase,**extra:progress(phase,iteration=trainer.iteration+1,**extra))
        except (FloatingPointError, AssertionError) as error:
            diagnostic={'error':str(error),'iteration':trainer.iteration+1,'updates':trainer.updates,
                        'sigma':model.std.numpy().tolist(),'learning_rate':trainer.lr,
                        'batch_shapes':{k:list(v.shape) for k,v in batch.items()}}
            (out/'ppo_failure.json').write_text(json.dumps(diagnostic,indent=2))
            raise
        assert np.isfinite(obs.numpy()).all()
        row=dict(iteration=trainer.iteration,samples=args.envs*args.steps,seconds=time.perf_counter()-ts,
                 resets=int(flatten(5).float().sum().item()),device='CL',
                 **stats)
        row=logger.finish_iteration(row,collection_end-ts,time.perf_counter()-collection_end,float(model.std.mean().item()))
        history.append(row)
        run={'experiment':args.exp_name,'num_obs':num_obs,'observation_schema':'flare_26','physics':args.physics,'envs':args.envs,'steps':args.steps,'environment_resume':'fresh reset','logger':logger.state(),
             'execution':{'microbatch_size':trainer.microbatch_size,'inference_chunk_size':trainer.inference_chunk_size,'row_lookup':'native_opencl_index_load','cpu_staging_threads':cpu_staging_threads,'jit':trainer.jit_enabled,'jit_adam':trainer.jit_adam,'rollout_jit':trainer.rollout_jit,'batch_realize':trainer.batch_realize,'host_dispatch':host_dispatch}}
        if args.physics=='genesis':run.update(environment_config=env.source_config,task_signature=env.signature)
        if trainer.iteration % save_interval==0 or trainer.iteration==target_iteration:
            progress('checkpoint',iteration=trainer.iteration)
            trainer.save(out/'checkpoint.npz',run)
        (out/'metrics.json').write_text(json.dumps(history,indent=2))
    logger.close()
    np.savez(out/'model.npz',**{k:v.numpy() for k,v in get_state_dict(model).items()})
    delta=float(np.abs(trainer.params[0].numpy()-first).max());assert delta>0
    result=dict(status='PASS',scope=args.physics+' PPO training smoke; convergence unverified',max_weight_change=delta,
                setup_seconds=setup_seconds,training_seconds=time.perf_counter()-started,arguments=vars(args),config=cfg,
                iteration=trainer.iteration,updates=trainer.updates)
    progress('completed',iteration=trainer.iteration,updates=trainer.updates)
    (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
