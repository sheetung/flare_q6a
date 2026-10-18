"""Experimental original FLARE Genesis physics -> OpenCL PPO bridge.

Genesis/Taichi Vulkan does physics; its PyTorch API uses CPU staging.
That staging and original FLARE observation/reward tensors are CPU work,
while PPO neural network training uses CL. This is NOT an all-GPU pipeline.
"""
import importlib
import sys
from pathlib import Path
import os
# Exclude llvmpipe: Vulkan success must mean actual Adreno hardware.
icd=os.environ.get('RL_VULKAN_ICD','/usr/share/vulkan/icd.d/freedreno_icd.json')
if not Path(icd).is_file():
    raise RuntimeError('Set RL_VULKAN_ICD to the board hardware Vulkan ICD JSON')
if 'lvp' in Path(icd).read_text().lower() or 'llvmpipe' in Path(icd).read_text().lower():
    raise RuntimeError('Software Vulkan ICD is not permitted')
os.environ['VK_DRIVER_FILES']=icd
os.environ['VK_ICD_FILENAMES']=os.environ['VK_DRIVER_FILES']
import gstaichi

ROOT=Path(__file__).resolve().parents[2]
# Map old Taichi import paths onto the Genesis-maintained fork for a probe.
for name in ('_kernels','lang','lang.util','lang.impl','types','types.primitive_types','lang.exception','_lib','_lib.core'):
    importlib.import_module('gstaichi.'+name)
for key,value in list(sys.modules.items()):
    if key=='gstaichi' or key.startswith('gstaichi.'):
        sys.modules['taichi'+key[len('gstaichi'):]]=value
if not hasattr(gstaichi._lib.core,'DataType'):
    gstaichi._lib.core.DataType=gstaichi._lib.core.DataTypeCxx
sys.path.insert(0,str(ROOT/'Genesis'))
sys.path.insert(0,str(ROOT/'Flare'))
import genesis as gs
import torch
import genesis.utils.misc as gs_misc
# Old Genesis patches Taichi private launch signatures; use public fork kernels.
gs_misc._tensor_to_ext_arr_fast=gstaichi._kernels.tensor_to_ext_arr
gs_misc._matrix_to_ext_arr_fast=gstaichi._kernels.matrix_to_ext_arr
import yaml
from tinygrad import Tensor
from envs.scenario1_env import HoverEnv
from flare_q6a.utils.metrics import FlareMetricsTap
from flare_q6a.utils.config import environment_config,task_signature
from genesis.utils.geom import quat_to_R
import utils.mixer as mixer

_original_mixer=mixer.QuadXMixer
class CpuStagingMixer(_original_mixer):
    def __init__(self,*args,**kwargs):
        kwargs['device']='cpu'
        super().__init__(*args,**kwargs)
mixer.QuadXMixer=CpuStagingMixer

class GenesisBridge:
    def __init__(self,n=4,show_viewer=False,rigid=False):
        # Relative paths in original FLARE assets require this working directory.
        import os
        if show_viewer and not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')):
            raise RuntimeError('-v requires a desktop display; run from a graphical desktop terminal')
        thread_limit=os.environ.get('RL_TORCH_THREADS')
        if thread_limit is not None:
            if int(thread_limit)<1:raise ValueError('RL_TORCH_THREADS must be positive')
            torch.set_num_threads(int(thread_limit))
        os.chdir(ROOT/'Flare')
        print('phase: Genesis Vulkan init',flush=True)
        gs.init(backend=gs.vulkan,logging_level='warning')
        assert gstaichi.lang.impl.current_cfg().arch==gstaichi.vulkan, 'CPU physics fallback forbidden'
        cfg=environment_config(rigid);self.source_config=environment_config(rigid);self.signature=task_signature(rigid)
        env_class=HoverEnv
        if rigid:
            from flare_q6a.envs.scenario1_rigid_env import HoverEnv as RigidWaypointEnv
            env_class=RigidWaypointEnv
        if show_viewer:cfg['env_cfg']['visualize_target']=True
        print(f'phase: original FLARE scene build (viewer={show_viewer})',flush=True)
        self.env=env_class(n,cfg['env_cfg'],cfg['obs_cfg'],cfg['reward_cfg'],cfg['command_cfg'],show_viewer=show_viewer,device='cpu')
        self.n=n
        print('phase: original FLARE reset',flush=True)
        self.env.reset()
        self._terminal_obs=None
        original_reset=self.env.reset_idx
        def capture_reset(indices):
            if len(indices):
                self._terminal_obs=self._buffer_obs().clone()
            original_reset(indices)
        self.env.reset_idx=capture_reset
        self.metrics_tap=FlareMetricsTap(self.env)
        self.last_step_info={}
    def _buffer_obs(self):
        e=self.env;sc=e.obs_scales
        return torch.cat([
            torch.clip(e.rel_pos[:,:2]*sc['rel_pos_xy'],-1,1),
            torch.clip(e.rel_pos[:,2:3]*sc['rel_pos_z'],-1,1),
            torch.clip(e.next_rel_pos[:,:2]*sc['rel_pos_xy'],-1,1),
            torch.clip(e.next_rel_pos[:,2:3]*sc['rel_pos_z'],-1,1),
            torch.clip(e.base_lin_vel[:,:2]*sc['lin_vel_xy'],-1,1),
            torch.clip(e.base_lin_vel[:,2:3]*sc['lin_vel_z'],-1,1),
            quat_to_R(e.base_quat).reshape(self.n,9),e.last_actions,
            torch.clip(e.cable_angle*sc.get('cable_angle',1.),-1,1),
            torch.clip(e.cable_angle_vel*sc.get('cable_angle_vel',.1),-1,1)],dim=1)
    def obs(self):
        return Tensor(self.env.get_observations().numpy().copy(),device='CL')
    def step(self,action):
        self._terminal_obs=None
        self.metrics_tap.begin_step()
        obs,_,reward,done,info=self.env.step(torch.from_numpy(action.numpy().copy()))
        self.last_step_info=self.metrics_tap.finish(reward,done)
        out=Tensor(obs.numpy().copy(),device='CL')
        terminal=out if self._terminal_obs is None else Tensor(self._terminal_obs.numpy().copy(),device='CL')
        timeout=(info['time_outs'].bool() & ~self.env.crash_condition)
        return out,Tensor(reward.numpy().copy(),device='CL'),Tensor(done.numpy().astype(bool),device='CL'),Tensor(timeout.numpy(),device='CL'),terminal

if __name__=='__main__':
    import json
    e=GenesisBridge(1)
    for _ in range(4):
        obs,reward,done,_,_=e.step(Tensor([[0.,0.,0.,0.]],device='CL'))
    print(json.dumps({'status':'PASS','physics':'Vulkan','ppo_device':'CL','staging':'CPU','obs_shape':obs.shape,'reward':reward.numpy().tolist()}))
