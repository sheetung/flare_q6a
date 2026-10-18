"""Experimental GPU surrogate; NOT Genesis flexible-cable physics."""
from pathlib import Path
import yaml
from tinygrad import Tensor

ROOT = Path(__file__).resolve().parents[2]

def cross(a, b):
    return Tensor.stack(a[:,1]*b[:,2]-a[:,2]*b[:,1], a[:,2]*b[:,0]-a[:,0]*b[:,2], a[:,0]*b[:,1]-a[:,1]*b[:,0], dim=1)

def unit(a):
    return a / (a.square().sum(1, keepdim=True)+1e-12).sqrt()

def rotation(q):
    w,x,y,z = [q[:,i] for i in range(4)]
    return Tensor.stack(1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y),
                        2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x),
                        2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y),dim=1).reshape(-1,3,3)

def angle(x, y):
    # atan2 via acos: preserves quadrant, avoids division by zero.
    a = (y/(x.square()+y.square()+1e-12).sqrt()).clip(-1,1).acos()
    return (x < 0).where(-a,a)

class CableEnv:
    """Massless, always-taut cable, point load, first-order body-rate response.

    Constraint tension may be negative: this surrogate is not valid for slack
    cable flight. No contact or motor mixer. Not suitable for real flight.
    """
    def __init__(self, n=32):
        cfg = yaml.safe_load((ROOT/'flare_q6a/config/envs/s1_env_config.yaml').read_text())
        self.cfg, self.n = cfg, n
        self.dt, self.length, self.mass, self.load_mass = .01, cfg['env_cfg']['cable_length'], .3, .068
        self.age = Tensor.zeros(n)
        self.p = Tensor([0.,0.,1.]).reshape(1,3).expand(n,3).contiguous().realize()
        self.v = Tensor.zeros(n,3)
        self.q = Tensor([1.,0.,0.,0.]).reshape(1,4).expand(n,4).contiguous().realize()
        self.omega = Tensor.zeros(n,3)
        self.s = Tensor([0.,0.,-1.]).reshape(1,3).expand(n,3).contiguous().realize()
        self.sd = Tensor.zeros(n,3)
        self.last_actions = Tensor.zeros(n,4)
        self.angles = Tensor.zeros(n,2)
        self.angle_vel = Tensor.zeros(n,2)
        self.target, self.next_target = self.sample_target(), self.sample_target()

    def sample_target(self):
        c = self.cfg['command_cfg']
        lo = Tensor([c[k][0] for k in ('pos_x_range','pos_y_range','pos_z_range')])
        hi = Tensor([c[k][1] for k in ('pos_x_range','pos_y_range','pos_z_range')])
        return (lo+Tensor.rand(self.n,3)*(hi-lo)).realize()

    def obs(self):
        sc = self.cfg['obs_cfg']['obs_scales']
        scale = Tensor([sc['rel_pos_xy'],sc['rel_pos_xy'],sc['rel_pos_z']])
        vs = Tensor([sc['lin_vel_xy'],sc['lin_vel_xy'],sc['lin_vel_z']])
        return Tensor.cat(((self.target-self.p)*scale).clip(-1,1),
            ((self.next_target-self.p)*scale).clip(-1,1),(self.v*vs).clip(-1,1),
            rotation(self.q).reshape(self.n,9),self.last_actions,
            (self.angles*sc['cable_angle']).clip(-1,1),
            (self.angle_vel*sc['cable_angle_vel']).clip(-1,1),dim=1).realize()

    def step(self, raw):
        a = raw.clip(-1,1).detach()
        old_error = self.target-self.p
        old_angles = self.angles
        rates = a[:,1:]*Tensor([15.,15.,5.])
        self.omega = (self.omega+(rates-self.omega)*(self.dt/.05)).realize()
        qw,qv = self.q[:,:1],self.q[:,1:]
        dq = Tensor.cat(-(qv*self.omega).sum(1,keepdim=True),qw*self.omega+cross(qv,self.omega),dim=1)
        qnew = self.q+dq*(.5*self.dt)
        self.q = (qnew/(qnew.square().sum(1,keepdim=True)+1e-12).sqrt()).realize()
        # Net tension enforces |load_position-drone_position| = length.
        thrust = (a[:,:1]+1)*.5*self.cfg['env_cfg']['TWR_max']*self.mass*9.81
        force = rotation(self.q)[:,:,2]*thrust
        tension = (-(self.s*force).sum(1,keepdim=True)/self.mass
                   + self.length*self.sd.square().sum(1,keepdim=True))/(1/self.mass+1/self.load_mass)
        gravity = Tensor([0.,0.,-9.81])
        acc = gravity+force/self.mass+tension*self.s/self.mass
        sdd = (-force/self.mass-tension*self.s*(1/self.mass+1/self.load_mass))/self.length
        self.v = (self.v+acc*self.dt).realize()
        self.p = (self.p+self.v*self.dt).realize()
        self.sd = (self.sd+sdd*self.dt).realize()
        self.s = unit(self.s+self.sd*self.dt).realize()
        self.sd = (self.sd-self.s*(self.sd*self.s).sum(1,keepdim=True)).realize()
        self.angles = Tensor.stack(angle(self.s[:,0],-self.s[:,2]),angle(self.s[:,1],-self.s[:,2]),dim=1).realize()
        self.angle_vel = ((self.angles-old_angles)/self.dt).realize()
        error = self.target-self.p
        rmat = rotation(self.q)
        cable_angle = (-(self.s*rmat[:,:,2]).sum(1)).clip(-1,1).acos()
        self.age = (self.age+1).realize()
        crash = (error[:,0].abs()>3)|(error[:,1].abs()>3)|(error[:,2].abs()>2)|(self.p[:,2]<.1)|(cable_angle>1.571)
        timeout = self.age>=1500
        done = crash|timeout
        # FLARE reward terms and dt-scaled coefficients; before reset.
        rc = self.cfg['reward_cfg']; rs=rc['reward_scales']
        yaw = angle(rmat[:,1,0],rmat[:,0,0])
        reward = ((old_error.square().sum(1)-error.square().sum(1))*rs['target']
            +(a-self.last_actions).square().sum(1)*rs['smooth']
            +(yaw.abs()*rc['yaw_lambda']).exp()*rs['yaw']
            +(self.omega.square().sum(1)+1e-12).sqrt()/3.14159*rs['angular']
            +crash.float()*rs['crash']
            +(cable_angle>75*3.14159/180).float()*(-rc['unsafe_angle_penalty'])*rs['cable_angle_safety'])*self.dt
        reward = reward.realize()
        self.last_actions = a.realize()
        terminal_obs = self.obs()
        advance = (error.square().sum(1)<.25)&(~done)
        self.target = advance.unsqueeze(1).where(self.next_target,self.target).realize()
        self.next_target = advance.unsqueeze(1).where(self.sample_target(),self.next_target).realize()
        mask = done.unsqueeze(1)
        for name, initial in [('p',[0.,0.,1.]),('v',[0.,0.,0.]),('q',[1.,0.,0.,0.]),('omega',[0.,0.,0.]),
                              ('s',[0.,0.,-1.]),('sd',[0.,0.,0.]),('last_actions',[0.,0.,0.,0.]),
                              ('angles',[0.,0.]),('angle_vel',[0.,0.])]:
            setattr(self,name,mask.where(Tensor(initial),getattr(self,name)).realize())
        self.age = done.where(0,self.age).realize()
        self.target = mask.where(self.sample_target(),self.target).realize()
        self.next_target = mask.where(self.sample_target(),self.next_target).realize()
        for name in ('p','v','q','omega','s','sd','last_actions','angles','angle_vel','age','target','next_target'):
            setattr(self,name,getattr(self,name).detach())
        return self.obs().detach(), reward.detach(), done.realize(), (timeout & (~crash)).realize(), terminal_obs.detach()
