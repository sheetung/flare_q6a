"""Single rigid rod hover environment; Vulkan physics, OpenCL policy interface."""
from pathlib import Path
import os,math
import xml.etree.ElementTree as ET
import numpy as np
import yaml
from tinygrad import Tensor
from genesis_bridge import gs,gstaichi,torch,ROOT
from hover_observation import rotation,pack_hover

class RigidHover:
    num_obs=19
    def __init__(self,n=8,show_viewer=False):
        self.n=n;self.cfg=yaml.safe_load((ROOT/'configs/rigid_hover.yaml').read_text())
        assert self.cfg['observation_size']==19 and self.cfg['observation_schema']=='hover_full_state_v1'
        assert self.cfg['dt']>0 and self.cfg['physics_substeps']>=1
        self.dt=self.cfg['dt'];self.length=self.cfg['rod_length'];self.target=np.array(self.cfg['payload_target'],np.float32)
        asset=ET.parse(ROOT/'assets/rigid_hover/drone_payload.urdf').getroot()
        assert abs(float(asset.find("joint[@name='payload_fixed']/origin").get('xyz').split()[2])+self.length)<1e-7,'config/URDF rod length mismatch'
        for link,key in [('base_link','drone_mass'),('end_sphere','payload_mass'),('rigid_rod','rod_mass')]:
            assert abs(float(asset.find(f"link[@name='{link}']/inertial/mass").get('value'))-self.cfg[key])<1e-7,'config/URDF mass mismatch'
        self.max_episode_length=math.ceil(self.cfg['episode_seconds']/self.dt)
        if show_viewer and not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')):
            raise RuntimeError('-v requires a graphical desktop terminal')
        print('phase: rigid hover Genesis Vulkan init',flush=True)
        gs.init(backend=gs.vulkan,logging_level='warning')
        assert gstaichi.lang.impl.current_cfg().arch==gstaichi.vulkan
        self.physics_steps=self.cfg['physics_substeps'];physics_dt=self.dt/self.physics_steps
        # This Genesis version clears external motor forces every physics substep.
        # Advance explicit 5 ms steps and reapply held RPM on each, rather than
        # applying force once before two internal substeps (half mean thrust).
        self.scene=gs.Scene(sim_options=gs.options.SimOptions(dt=physics_dt,substeps=1),
            rigid_options=gs.options.RigidOptions(dt=physics_dt,enable_collision=False,enable_joint_limit=False),
            viewer_options=gs.options.ViewerOptions(camera_pos=(2.,3.,2.),camera_lookat=(0.,0.,.7),max_FPS=60),
            vis_options=gs.options.VisOptions(rendered_envs_idx=[0]),show_viewer=show_viewer)
        self.scene.add_entity(gs.morphs.Plane())
        self.marker=None
        if show_viewer:
            self.marker=self.scene.add_entity(gs.morphs.Sphere(radius=.04,pos=tuple(self.target),fixed=True,collision=False),
                surface=gs.surfaces.Rough(color=(.2,.8,.2)))
        self.drone=self.scene.add_entity(gs.morphs.Drone(file=str(ROOT/'assets/rigid_hover/drone_payload.urdf'),links_to_keep=['end_sphere']))
        print('phase: rigid hover scene build',flush=True)
        self.scene.build(n_envs=n)
        self.payload=self.drone.get_link('end_sphere')
        self.pivot_dofs=[];self.pivot_qs=[]
        self._all_envs=torch.arange(n,dtype=gs.tc_int)
        for name in ('pivot_roll','pivot_pitch','pivot_yaw'):
            joint=self.drone.get_joint(name)
            self.pivot_dofs.extend(joint.dofs_idx_local);self.pivot_qs.extend(joint.qs_idx_local)
        assert len(self.pivot_dofs)==3 and self.drone.n_qs==10
        self.age=np.zeros(n,np.int32);self.last_actions=np.zeros((n,4),np.float32)
        self.rng=np.random.default_rng(1);self.last_step_info={}
        self.reset(np.arange(n));self._read_state();self._obs=self._observation()
    @staticmethod
    def _tensor(a):return torch.as_tensor(np.asarray(a),dtype=torch.float32)
    def reset(self,ids):
        ids=np.asarray(ids,dtype=np.int32)
        if not len(ids):return
        pos=np.tile(self.target+[0,0,self.length],(len(ids),1)).astype(np.float32)
        pos+=self.rng.uniform(-self.cfg['reset_position_jitter'],self.cfg['reset_position_jitter'],pos.shape).astype(np.float32)
        q=np.tile([1.,0.,0.,0.],(len(ids),1)).astype(np.float32)
        angles=self.rng.uniform(-self.cfg['reset_swing_radians'],self.cfg['reset_swing_radians'],(len(ids),3)).astype(np.float32)
        # Write generalized coordinates atomically. Old Genesis partial-FK kernel
        # iterates ndarray indices rather than selected environment IDs. Calling
        # individual pose/joint setters could restore stale root coordinates.
        qpos=np.zeros((len(ids),self.drone.n_qs),np.float32)
        qpos[:,:3]=pos;qpos[:,3:7]=q;qpos[:,self.pivot_qs]=angles
        self.drone.set_qpos(self._tensor(qpos),zero_velocity=True,envs_idx=ids.tolist())
        # Full contiguous refresh works around that pinned upstream kernel bug;
        # non-reset environments retain their qpos and generalized velocities.
        self.drone.solver._kernel_forward_kinematics_links_geoms(self._all_envs)
        self.age[ids]=0;self.last_actions[ids]=0
    def _read_state(self):
        def a(x):return x.detach().cpu().numpy().copy()
        self.pos=a(self.drone.get_pos());self.vel=a(self.drone.get_vel());self.quat=a(self.drone.get_quat())
        self.r=rotation(self.quat)
        self.omega=np.einsum('nji,nj->ni',self.r,a(self.drone.get_ang()))
        self.load=a(self.payload.get_pos());self.load_vel=a(self.payload.get_vel())
    def _observation(self):
        return pack_hover(self.pos,self.vel,self.quat[:,[1,2,3,0]],self.omega,self.load,self.load_vel,self.target,self.length)
    def obs(self):return Tensor(self._obs.copy(),device='CL')
    def step(self,action):
        act=np.clip(action.numpy(),-1,1).astype(np.float32)
        rate=act[:,1:]*np.array(self.cfg['max_rate'])
        # Proportional body-rate controller, independent of policy observation.
        torques=.1*(rate-self.omega)
        mix=np.array([[-1,-1,-1],[-1,1,1],[1,1,-1],[1,-1,1]],np.float32)
        motor=np.clip((act[:,:1]+1)/2+torques@mix.T,0,1)
        mass=sum(self.cfg[k] for k in ['drone_mass','payload_mass','rod_mass','pivot_mass'])
        max_force=mass*9.81*self.cfg['thrust_to_weight']
        rpm=np.sqrt(motor*max_force/(4*self.drone._KF)).astype(np.float32)
        for _ in range(self.physics_steps):
            self.drone.set_propellels_rpm(self._tensor(rpm));self.scene.step()
        self.age+=1
        self._read_state();terminal=self._observation()
        rope=self.load-self.pos;distance=np.linalg.norm(self.load-self.target,axis=1)
        measured_length=np.linalg.norm(rope,axis=1)
        swing=np.arccos(np.clip(-rope[:,2]/np.maximum(measured_length,1e-8),-1,1))
        tilt=np.arccos(np.clip(self.r[:,2,2],-1,1))
        causes={'ground':(self.load[:,2]<.08)|(self.pos[:,2]<.1),
                'position_limit':np.linalg.norm(self.pos-(self.target+[0,0,self.length]),axis=1)>2.,
                'tilt_limit':tilt>np.deg2rad(75),'swing_limit':swing>np.deg2rad(80)}
        crash=np.logical_or.reduce(list(causes.values()));timeout=(self.age>=self.max_episode_length)&~crash;done=crash|timeout
        raw={'hover':np.exp(-(distance/.25)**2),'payload_velocity':np.sum(self.load_vel**2,axis=1),
             'swing':np.sum(rope[:,:2]**2,axis=1)/self.length**2,'tilt':1-self.r[:,2,2],
             'action_change':np.sum((act-self.last_actions)**2,axis=1),'crash':crash.astype(np.float32)}
        parts={k:v*self.cfg['reward'][k]*(1 if k=='crash' else self.dt) for k,v in raw.items()}
        reward=sum(parts.values()).astype(np.float32)
        success=(distance<.1)&(np.linalg.norm(self.load_vel,axis=1)<.2)&(swing<np.deg2rad(10))
        self.last_step_info=dict(raw_rewards=reward.copy(),dones=done.copy(),reward_parts=parts,
            drone_error_m=np.linalg.norm(self.pos-(self.target+[0,0,self.length]),axis=1),payload_error_m=distance,
            swing_deg=np.rad2deg(swing),rope_body_deg=np.rad2deg(np.arccos(np.clip(-np.einsum('ni,ni->n',rope,self.r[:,:,2])/np.maximum(measured_length,1e-8),-1,1))),
            crashes=crash,timeouts_only=timeout,causes=causes,hover_success=success.astype(float),rod_length_error_m=np.abs(measured_length-self.length))
        self.last_actions[:]=act
        self.reset(np.flatnonzero(done))
        if np.any(done):self._read_state()
        self._obs=self._observation()
        def cl(x):return Tensor(np.asarray(x).copy(),device='CL')
        return cl(self._obs),cl(reward),cl(done),cl(timeout),cl(terminal)
