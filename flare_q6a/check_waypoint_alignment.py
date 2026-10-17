"""Check copied task methods against pinned FLARE; optional real Vulkan tests."""
import argparse,ast,json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from task_config import ROOT,MODEL,DEFAULT_ENVS,environment_config,train_config
p=argparse.ArgumentParser();p.add_argument('--physics',action='store_true');a=p.parse_args()
def methods(path):
 tree=ast.parse(path.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='HoverEnv')
 return {f.name:ast.dump(f,include_attributes=False) for f in cls.body if isinstance(f,ast.FunctionDef)}
old=methods(ROOT/'Flare/envs/scenario1_env.py');new=methods(ROOT/'flare_q6a/rigid_waypoint_env.py')
assert old.keys()==new.keys()
changed=[k for k in old if old[k]!=new[k]];assert sorted(changed)==['__init__','reset_idx'],changed
original=environment_config(False);rigid=environment_config(True)
expected=json.loads(json.dumps(original));expected['env_cfg']['num_of_dofs']=3;assert rigid==expected
cfg=train_config();assert cfg['runner']['num_steps_per_env']==100 and cfg['algorithm']['num_mini_batches']==4
assert DEFAULT_ENVS==4 and DEFAULT_ENVS*100//4==100
model=ET.parse(MODEL).getroot();source=ET.parse(ROOT/'Flare/assets/suspended_system/real_sus_exp.urdf').getroot()
def mass(r):return sum(float(x.get('value')) for x in r.findall('link/inertial/mass'))
assert abs(mass(source)-mass(model))<1e-10
joints=[j for j in model.findall('joint') if j.get('type')!='fixed'];assert len(joints)==3
assert all(j.find('origin').get('xyz')=='0 0 0' for j in joints)
assert model.find("joint[@name='payload_fixed']").get('type')=='fixed'
checks=['all step/reward/target/observation/safety methods AST-identical to FLARE',
        'only model construction and articulated reset differ','only num_of_dofs differs in environment configuration',
        'original PPO/100-step rollout, parallel 4 -> minibatch 100','total physical mass matches original model','one rigid rod, root free rotation and fixed payload']
if a.physics:
 from tinygrad import Tensor
 from genesis_bridge import GenesisBridge,torch
 env=GenesisBridge(4,rigid=True);e=env.env
 for _ in range(12):
  o,r,d,to,t=env.step(Tensor(np.zeros((4,4),np.float32),device='CL'))
  assert o.shape==(4,26) and np.isfinite(o.numpy()).all() and np.isfinite(t.numpy()).all()
  np.testing.assert_allclose(sum(env.last_step_info['reward_parts'].values()),r.numpy(),atol=1e-6)
  np.testing.assert_allclose(np.linalg.norm(e.drone.get_link('end_sphere').get_pos().numpy()-e.drone.get_pos().numpy(),axis=1),.48,atol=5e-5)
 for ids in [[3],[1,3],[0,3]]:
  q=e.drone.get_qpos().numpy().copy();v=e.drone.get_dofs_velocity().numpy().copy()
  e.reset_idx(torch.tensor(ids,dtype=torch.int32))
  np.testing.assert_allclose(e.drone.get_pos().numpy()[ids],np.tile(e.base_init_pos.numpy(),(len(ids),1)),atol=1e-6)
  np.testing.assert_allclose(e.drone.get_link('end_sphere').get_pos().numpy()[ids],np.tile(e.payload_init_pos.numpy(),(len(ids),1)),atol=1e-6)
  np.testing.assert_allclose(e.drone.get_vel().numpy()[ids],0,atol=1e-6)
  np.testing.assert_allclose(e.drone.get_link('end_sphere').get_vel().numpy()[ids],0,atol=1e-6)
  other=np.setdiff1d(np.arange(4),ids)
  np.testing.assert_allclose(e.drone.get_qpos().numpy()[other],q[other],atol=1e-7)
  np.testing.assert_allclose(e.drone.get_dofs_velocity().numpy()[other],v[other],atol=1e-7)
 # Exercise original waypoint acceptance radius and next-target transition.
 e.commands[:]=e.base_pos;e.rel_pos=e.commands-e.base_pos;e.next_commands[:]=torch.tensor([.7,-.4,1.1])
 hits=e._at_target();assert len(hits)==4;e._resample_commands(hits)
 np.testing.assert_allclose(e.commands.numpy(),np.tile([.7,-.4,1.1],(4,1)),atol=1e-6)
 e.max_episode_length=2;e.episode_length_buf[:]=0
 timeout_count=0
 for _ in range(4):
  o,r,d,to,t=env.step(Tensor(np.zeros((4,4),np.float32),device='CL'));timeout_count+=int(to.numpy().sum())
 assert timeout_count>0
 checks+=['actual Vulkan 26-D observation/reward decomposition/rigid length',
          'non-contiguous reset clears selected state and preserves others','original target transition and timeout']
print(json.dumps(dict(status='PASS',total_mass_kg=mass(model),changed_methods=changed,checks=checks)))
