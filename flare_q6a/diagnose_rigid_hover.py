"""Trace fixed-hover and seeded exploration, including state before/after reset."""
import argparse,json
from pathlib import Path
import numpy as np
from tinygrad import Tensor
from rigid_hover import RigidHover
p=argparse.ArgumentParser();p.add_argument('--envs',type=int,default=16);p.add_argument('--steps',type=int,default=100);p.add_argument('--mode',choices=['hover','random'],default='hover');p.add_argument('--output',required=True);a=p.parse_args()
e=RigidHover(a.envs);rng=np.random.default_rng(123);rows=[];age=np.zeros(a.envs,int);lengths=[];first=[]
for i in range(a.steps):
 before=e.obs().numpy();age+=1
 act=np.tile([-.2,0,0,0],(a.envs,1)).astype(np.float32)
 if a.mode=='random':act=rng.normal(0,1,(a.envs,4)).astype(np.float32)
 o,r,d,to,t=e.step(Tensor(act,device='CL'));term=t.numpy();done=d.numpy();after=o.numpy()
 info=e.last_step_info
 row=dict(step=i+1,done_count=int(done.sum()),max_speed=float(np.linalg.norm(term[:,3:6]*2,axis=1).max()),
  max_omega=float(np.linalg.norm(term[:,10:13]*5,axis=1).max()),max_position_error=float(np.linalg.norm(term[:,:3],axis=1).max()),
  max_swing=float(info['swing_deg'].max()),max_rod_error=float(info['rod_length_error_m'].max()),
  causes={k:int(v.sum()) for k,v in info['causes'].items()})
 if done.any():
  lengths.extend(age[done].tolist());age[done]=0
  k=int(np.flatnonzero(done)[0]);row['first_failure_env']=k;row['before']=before[k].tolist();row['terminal']=term[k].tolist();row['after_reset']=after[k].tolist();row['action']=act[k].tolist()
  if len(first)<4:first.append(row.copy())
 rows.append(row)
 if i<12 or done.any():print(json.dumps(row),flush=True)
result=dict(mode=a.mode,envs=a.envs,steps=a.steps,resets=len(lengths),episode_lengths=lengths,
 first_failures=first,max_speed=max(r['max_speed'] for r in rows),max_omega=max(r['max_omega'] for r in rows),rows=rows)
out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ['rows','first_failures','episode_lengths']}),flush=True)
