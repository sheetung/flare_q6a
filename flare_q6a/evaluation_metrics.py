"""Episode-level policy evaluation with censored trajectories kept separate."""
import numpy as np

class EvaluationMetrics:
    def __init__(self,n,dt,episodes=None):
        self.n=n;self.dt=dt;self.quota=None if episodes is None else np.array([episodes//n+(i<episodes%n) for i in range(n)])
        self.completed=np.zeros(n,int);self.length=np.zeros(n,int);self.returns=np.zeros(n)
        self.sums={};self.peaks={};self.rows=[];self.total_samples=0;self.clipped=0;self.action_components=0
    def active(self):return np.ones(self.n,bool) if self.quota is None else self.completed<self.quota
    def observe(self,reward,done,timeout,info,action):
        active=self.active();reward=np.asarray(reward);done=np.asarray(done,dtype=bool);timeout=np.asarray(timeout,dtype=bool)
        self.length[active]+=1;self.returns[active]+=reward[active];self.total_samples+=int(active.sum())
        self.clipped+=int((np.abs(action[active])>1).sum());self.action_components+=int(action[active].size)
        for key in ('drone_error_m','payload_error_m','swing_deg','hover_success','rod_length_error_m'):
            if key in info:
                value=np.asarray(info[key]);self.sums.setdefault(key,np.zeros(self.n));self.sums[key][active]+=value[active]
                self.peaks.setdefault(key,np.zeros(self.n));self.peaks[key][active]=np.maximum(self.peaks[key][active],value[active])
        new=[]
        for i in np.flatnonzero(active & done):
            row=dict(env_id=int(i),episode_index=int(self.completed[i]+1),steps=int(self.length[i]),
                     seconds=float(self.length[i]*self.dt),reward=float(self.returns[i]),timeout_only=bool(timeout[i]),
                     failed=bool(np.asarray(info.get('crashes',done & ~timeout))[i]),
                     causes=[k for k,v in info.get('causes',{}).items() if np.asarray(v)[i]])
            row.update({'mean_'+k:float(v[i]/self.length[i]) for k,v in self.sums.items()})
            row.update({'peak_'+k:float(v[i]) for k,v in self.peaks.items() if k in ('swing_deg','rod_length_error_m')})
            self.rows.append(row);new.append(row);self.completed[i]+=1;self.length[i]=0;self.returns[i]=0
            for v in self.sums.values():v[i]=0
            for v in self.peaks.values():v[i]=0
        return new
    def summary(self):
        rows=self.rows;durations=[r['seconds'] for r in rows];steps=sum(r['steps'] for r in rows)
        partial=[]
        for i in np.flatnonzero(self.length>0):
            partial.append(dict(env_id=int(i),steps=int(self.length[i]),seconds=float(self.length[i]*self.dt),reward=float(self.returns[i]),censored=True))
        result=dict(completed_episodes=len(rows),requested_episodes=int(self.quota.sum()) if self.quota is not None else None,
            requested_episodes_complete=bool(not self.active().any()) if self.quota is not None else None,
            evaluated_samples=self.total_samples,failures=sum(r['failed'] for r in rows),
            safe_full_duration_episodes=sum(r['timeout_only'] for r in rows),
            mean_episode_seconds=float(np.mean(durations)) if rows else None,
            median_episode_seconds=float(np.median(durations)) if rows else None,
            max_episode_seconds=max(durations) if rows else None,
            mean_episode_reward=float(np.mean([r['reward'] for r in rows])) if rows else None,
            action_clip_fraction=self.clipped/max(1,self.action_components),
            termination_causes={k:sum(k in r['causes'] for r in rows) for k in sorted({k for r in rows for k in r['causes']})},
            partial_episodes=partial,
            note='Task means are step-weighted over completed episodes; partial episodes are censored. Safe full duration alone does not establish precise hover.')
        for key in self.sums:
            result['mean_'+key]=sum(r.get('mean_'+key,0)*r['steps'] for r in rows)/steps if steps else None
        if rows and any('peak_swing_deg' in r for r in rows):result['peak_swing_deg']=max(r.get('peak_swing_deg',0) for r in rows)
        return result
