"""Meaningful topology, observation, rigid constraint and reset checks."""
import argparse,json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from hover_observation import pack_hover,rotation
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--physics',action='store_true');a=p.parse_args()
r=ET.parse(ROOT/'assets/rigid_hover/drone_payload.urdf').getroot()
movable=[j for j in r.findall('joint') if j.get('type')!='fixed']
assert len(movable)==3
assert {j.get('name') for j in movable}=={'pivot_roll','pivot_pitch','pivot_yaw'}
assert all(j.find('origin').get('xyz')=='0 0 0' for j in movable)
j=r.find("joint[@name='payload_fixed']");assert j.get('type')=='fixed' and j.find('origin').get('xyz')=='0 0 -0.48'
target=np.array([0,0,.52]);pos=target+[0,0,.48]
obs=pack_hover(pos,[0,0,0],[0,0,0,1],[0,0,0],target,[0,0,0],target)
assert obs.shape==(19,);expected=np.zeros(19);expected[9]=1
np.testing.assert_allclose(obs,expected,atol=1e-7)
np.testing.assert_allclose(obs,pack_hover(pos,[0,0,0],[0,0,0,-1],[0,0,0],target,[0,0,0],target))
np.testing.assert_allclose(rotation(np.array([1.,0,0,0])),np.eye(3))
checks=['three colocated root pivot rotations only','one rod and fixed payload joint','19-D observation order/scales and quaternion sign']
if a.physics:
 from tinygrad import Tensor
 from rigid_hover import RigidHover
 env=RigidHover(2);env.cfg['reset_position_jitter']=0.;env.cfg['reset_swing_radians']=0.
 def reset():
  env.reset(np.arange(2));env._read_state();env._obs=env._observation()
 reset();start=env.load.copy();max_length_error=0.
 hover=np.tile([2/env.cfg['thrust_to_weight']-1,0,0,0],(2,1)).astype(np.float32)
 for _ in range(20):
  o,r,d,to,t=env.step(Tensor(hover,device='CL'))
  assert not d.numpy().any()
  max_length_error=max(max_length_error,float(env.last_step_info['rod_length_error_m'].max()))
  np.testing.assert_allclose(sum(env.last_step_info['reward_parts'].values()),r.numpy(),atol=1e-7)
 print(json.dumps({'hover_start':start.tolist(),'hover_end':env.load.tolist(),'base_end':env.pos.tolist(),'base_velocity':env.vel.tolist(),'quaternion':env.quat.tolist(),'rod_error':max_length_error}),flush=True)
 assert np.linalg.norm(env.load-start,axis=1).max()<.03,'nominal thrust does not balance weight'
 for _ in range(12):
  command=hover.copy();command[:,1]=.1
  o,r,d,to,t=env.step(Tensor(command,device='CL'))
  max_length_error=max(max_length_error,float(env.last_step_info['rod_length_error_m'].max()))
  assert np.isfinite(o.numpy()).all() and np.isfinite(t.numpy()).all()
 assert max_length_error<5e-4
 reset();env.max_episode_length=3
 for k in range(3):o,r,d,to,t=env.step(Tensor(hover,device='CL'))
 assert d.numpy().all() and to.numpy().all() and (env.age==0).all()
 env.max_episode_length=1500;reset()
 env.drone.set_pos(env._tensor(np.tile([0,0,.05],(2,1))),zero_velocity=True)
 o,r,d,to,t=env.step(Tensor(hover,device='CL'))
 assert d.numpy().all() and not to.numpy().any()
 assert env.last_step_info['causes']['ground'].all()
 assert not np.allclose(o.numpy(),t.numpy()),'terminal observation lost during reset'
 checks+=['Vulkan rigid distance during hover and rate command','nominal thrust balances coupled mass',
          'exact reward decomposition','timeout exclusive of crash','pre-reset terminal observation preserved']
 print(json.dumps(dict(status='PASS',max_rod_length_error_m=max_length_error,checks=checks)))
else:print(json.dumps(dict(status='PASS',checks=checks)))
