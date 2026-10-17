"""Non-mutating FLARE instrumentation and rsl_rl-style experiment logging."""
from collections import deque
from datetime import datetime,timezone
import json,time
from pathlib import Path
import numpy as np

def array(x):
    if hasattr(x,'detach'):x=x.detach()
    if hasattr(x,'cpu'):x=x.cpu()
    if hasattr(x,'numpy'):x=x.numpy()
    return np.asarray(x).copy()

class FlareMetricsTap:
    """Observe terminal state before reset; record exact rewards when computed."""
    def __init__(self,env):
        self.env=env;self.begin_step()
        reset=env.reset_idx
        def observed_reset(indices):
            self.snapshot=self.capture()
            return reset(indices)
        env.reset_idx=observed_reset
        at_target=env._at_target
        def observed_target():
            ids=at_target();self.hits=array(ids).astype(int);return ids
        env._at_target=observed_target
        for name,func in list(env.reward_functions.items()):
            def observed_reward(func=func,name=name):
                value=func()
                self.parts[name]=array(value)*env.reward_scales[name]
                return value
            env.reward_functions[name]=observed_reward
    def begin_step(self):self.parts={};self.snapshot={};self.hits=np.array([],dtype=int)
    def capture(self):
        e=self.env;c=e.env_cfg
        pos=array(e.base_pos);payload=array(e.payload_pos);angle=array(e.cable_body_angle)
        rel=array(e.rel_pos);euler=array(e.base_euler);age=array(e.episode_length_buf)
        rope=payload-pos;length=np.linalg.norm(rope,axis=1)
        cos=np.divide(-rope[:,2],length,out=np.ones_like(length),where=length>1e-9)
        causes={
            'roll_limit':np.abs(euler[:,0])>c['termination_if_roll_greater_than'],
            'pitch_limit':np.abs(euler[:,1])>c['termination_if_pitch_greater_than'],
            'x_limit':np.abs(rel[:,0])>c['termination_if_x_greater_than'],
            'y_limit':np.abs(rel[:,1])>c['termination_if_y_greater_than'],
            'z_limit':np.abs(rel[:,2])>c['termination_if_z_greater_than'],
            'ground':pos[:,2]<c['termination_if_close_to_ground'],
            'safety_count':array(e.safety_violation_count)>e.max_safety_violations,
            'rope_body_angle':angle>c.get('termination_cable_angle_threshold',np.pi/2)}
        crash=array(e.crash_condition).astype(bool)
        timeout=age>e.max_episode_length
        return {'drone_error_m':np.linalg.norm(rel,axis=1),
            'payload_error_m':np.linalg.norm(array(e.rel_pos_payload),axis=1),
            'swing_deg':np.degrees(np.arccos(np.clip(cos,-1,1))),
            'rope_body_deg':np.degrees(angle),'crashes':crash,
            'timeouts_only':timeout & ~crash,'causes':causes,
            'waypoint_hits':np.bincount(self.hits,minlength=e.num_envs)}
    def finish(self,reward,done):
        return dict(self.snapshot,raw_rewards=array(reward),dones=array(done).astype(bool),reward_parts=dict(self.parts))

class ExperimentLogger:
    def __init__(self,out,n,dt,target_iteration,initial_samples=0,initial_time=0):
        from torch.utils.tensorboard import SummaryWriter
        self.out=Path(out);self.n=n;self.dt=dt;self.target=target_iteration
        self.writer=SummaryWriter(str(self.out/'tensorboard'))
        self.returns=np.zeros(n);self.lengths=np.zeros(n,dtype=int);self.ep_hits=np.zeros(n,dtype=int)
        self.ep_parts={};self.episodes=deque(maxlen=100)
        self.total_samples=initial_samples;self.total_time=initial_time;self.times=deque(maxlen=20)
        self.begin_iteration()
    def begin_iteration(self):
        self.count=0;self.roll_sum=0.;self.part_sums={};self.hits=0;self.done_count=0;self.crashes=0;self.timeouts=0
        self.causes={};self.measures={};self.swing_max=None
    def observe(self,reward,done,info=None):
        info=info or {};r=array(info['raw_rewards']) if 'raw_rewards' in info else array(reward)
        d=array(info['dones']).astype(bool) if 'dones' in info else array(done).astype(bool)
        self.returns+=r;self.lengths+=1;self.count+=len(r);self.roll_sum+=float(r.sum());self.done_count+=int(d.sum())
        hits=np.asarray(info.get('waypoint_hits',np.zeros(self.n)),dtype=int)
        self.ep_hits+=hits;self.hits+=int(hits.sum())
        for key,v in info.get('reward_parts',{}).items():
            v=np.asarray(v);self.part_sums[key]=self.part_sums.get(key,0.)+float(v.sum())
            self.ep_parts.setdefault(key,np.zeros(self.n));self.ep_parts[key]+=v
        for key in ('drone_error_m','payload_error_m','swing_deg','rope_body_deg'):
            if key in info:
                v=np.asarray(info[key]);finite=v[np.isfinite(v)];s,c=self.measures.get(key,(0.,0))
                self.measures[key]=(s+float(finite.sum()),c+len(finite))
                if key=='swing_deg' and len(finite):self.swing_max=max(self.swing_max or 0.,float(finite.max()))
        self.crashes+=int(np.asarray(info.get('crashes',[])).sum())
        self.timeouts+=int(np.asarray(info.get('timeouts_only',[])).sum())
        for key,v in info.get('causes',{}).items():self.causes[key]=self.causes.get(key,0)+int(np.asarray(v).sum())
        for i in np.flatnonzero(d):
            self.episodes.append({'reward':float(self.returns[i]),'length':int(self.lengths[i]),'waypoints':int(self.ep_hits[i]),
                                  'parts':{k:float(v[i]) for k,v in self.ep_parts.items()}})
            self.returns[i]=0.;self.lengths[i]=0;self.ep_hits[i]=0
            for v in self.ep_parts.values():v[i]=0.
    def finish_iteration(self,row,collection_seconds,learning_seconds,noise_std):
        self.total_samples+=row['samples'];self.total_time+=row['seconds'];self.times.append(row['seconds'])
        ep=list(self.episodes)
        row.update(utc=datetime.now(timezone.utc).isoformat(),collection_seconds=collection_seconds,learning_seconds=learning_seconds,
            fps=row['samples']/max(row['seconds'],1e-9),total_samples=self.total_samples,total_training_seconds=self.total_time,
            eta_seconds=float(np.mean(self.times))*max(0,self.target-row['iteration']),mean_action_noise_std=float(noise_std),
            mean_raw_reward=self.roll_sum/max(1,self.count),completed_episodes_this_iteration=self.done_count,
            episode_window_count=len(ep),mean_episode_reward=float(np.mean([e['reward'] for e in ep])) if ep else None,
            mean_episode_length=float(np.mean([e['length'] for e in ep])) if ep else None,
            waypoints_reached=self.hits,waypoints_per_1000_steps=1000*self.hits/max(1,self.count),
            crashes=self.crashes,timeouts_only=self.timeouts,termination_causes=self.causes,
            reward_terms_per_step={k:v/max(1,self.count) for k,v in self.part_sums.items()},
            episode_reward_terms={k:float(np.mean([e['parts'][k] for e in ep if k in e['parts']])) for k in self.ep_parts} if ep else {},
            peak_swing_deg=self.swing_max)
        for k,(s,c) in self.measures.items():row['mean_'+k]=s/c if c else None
        tags={'Loss/value_function':'value_loss','Loss/surrogate':'policy_loss','Loss/learning_rate':'learning_rate',
              'Policy/mean_noise_std':'mean_action_noise_std','Policy/KL':'kl','Policy/entropy':'entropy',
              'Perf/total_fps':'fps','Perf/collection_time':'collection_seconds','Perf/learning_time':'learning_seconds',
              'Train/mean_reward':'mean_episode_reward','Train/mean_episode_length':'mean_episode_length',
              'Train/mean_step_reward':'mean_raw_reward','Task/waypoints_reached':'waypoints_reached',
              'Task/drone_error_m':'mean_drone_error_m','Task/payload_error_m':'mean_payload_error_m',
              'Task/swing_deg':'mean_swing_deg','Task/peak_swing_deg':'peak_swing_deg',
              'Termination/crashes':'crashes','Termination/timeouts':'timeouts_only'}
        for tag,key in tags.items():
            if row.get(key) is not None:self.writer.add_scalar(tag,row[key],row['iteration'])
        for k,v in row['reward_terms_per_step'].items():self.writer.add_scalar('RewardPerStep/'+k,v,row['iteration'])
        for k,v in row['episode_reward_terms'].items():self.writer.add_scalar('Episode/return_'+k,v,row['iteration'])
        for k,v in self.causes.items():self.writer.add_scalar('Termination/'+k,v,row['iteration'])
        self.writer.flush()
        with (self.out/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(self.format(row),flush=True)
        return row
    def format(self,r):
        def fmt(v,d=4):return 'N/A (no completed episodes yet)' if v is None else f'{v:.{d}f}'
        lines=['#'*80,f"Learning iteration {r['iteration']}/{self.target}".center(80),'']
        def add(label,value):lines.append(f'{label+":":>36} {value}')
        add('Computation',f"{r['fps']:.1f} steps/s (collection {r['collection_seconds']:.2f}s, learning {r['learning_seconds']:.2f}s)")
        for label,key in [('Value function loss','value_loss'),('Surrogate loss','policy_loss'),('KL','kl'),('Mean action noise std','mean_action_noise_std')]:add(label,fmt(r[key]))
        add('Learning rate',f"{r['learning_rate']:.3e}")
        add('Completed episode window',r['episode_window_count'])
        add('Mean reward (last 100 episodes)',fmt(r['mean_episode_reward']))
        length=r['mean_episode_length'];add('Mean episode length', 'N/A' if length is None else f'{length:.1f} steps / {length*self.dt:.2f}s')
        add('Mean reward per step',fmt(r['mean_raw_reward'],6))
        add('Episodes completed this iteration',r['completed_episodes_this_iteration'])
        add('Crash / timeout-only resets',f"{r['crashes']} / {r['timeouts_only']}")
        add('Waypoints reached this iteration',r['waypoints_reached'])
        for label,key in [('Mean drone target error (m)','mean_drone_error_m'),('Mean payload target error (m)','mean_payload_error_m'),('Mean swing from vertical (deg)','mean_swing_deg'),('Peak swing from vertical (deg)','peak_swing_deg'),('Mean rope/body angle (deg)','mean_rope_body_deg')]:
            if key in r:add(label,fmt(r[key],3))
        for k,v in r['episode_reward_terms'].items():add('Mean episode reward/'+k,fmt(v,6))
        for k,v in r['reward_terms_per_step'].items():add('Mean step reward/'+k,fmt(v,6))
        add('Termination causes (may overlap)',', '.join(f'{k}={v}' for k,v in self.causes.items() if v) or 'none')
        try:
            t=json.loads((self.out/'latest_telemetry.json').read_text());u=t.get('gpu_engine_utilization_percent')
            if u is not None:add('GPU process engine (last window)',f'{u:.1f}%')
        except (OSError,ValueError):pass
        lines.append('-'*80);add('Total timesteps',r['total_samples']);add('Total training time',f"{r['total_training_seconds']:.1f}s")
        add('ETA (recent iterations)',f"{r['eta_seconds']/3600:.2f} hours");add('Checkpoint',str(self.out/'checkpoint.npz'))
        return '\n'.join(lines)+'\n'
    def state(self):return {'total_samples':self.total_samples,'total_time':self.total_time,'episodes':list(self.episodes)}
    def restore_episodes(self,state):self.episodes.extend(state.get('episodes',[]))
    def close(self):self.writer.close()
