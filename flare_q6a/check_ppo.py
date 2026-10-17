"""Numerical reference checks for GPU probability and clipping operations."""
import json, math
import numpy as np
from tinygrad import Tensor,Device,Context
from train import ActorCritic
assert Device.DEFAULT=='CL'
m=ActorCritic()
a=np.array([[.1,.2,-.3,.4],[-1.,.3,.5,.7]],np.float32)
mu=np.array([[.2,0.,-.1,.3],[.1,.2,.3,.4]],np.float32)
logstd=np.array([-.2,.1,-.1,.3],np.float32)
m.std.assign(Tensor(np.exp(logstd))).realize()
ref=(-.5*((a-mu)/np.exp(logstd))**2-logstd-.5*math.log(2*math.pi)).sum(-1)
np.testing.assert_allclose(m.logprob(Tensor(a),Tensor(mu)).numpy(),ref,atol=2e-5)
ratio=np.array([.5,.9,1.1,1.5],np.float32);adv=np.array([-1,1,-1,1],np.float32)
actual=(Tensor(ratio)*Tensor(adv)).minimum(Tensor(ratio).clip(.8,1.2)*Tensor(adv)).numpy()
np.testing.assert_allclose(actual,np.minimum(ratio*adv,np.clip(ratio,.8,1.2)*adv),atol=1e-6)
# GAE example, reset at index 1 must break recursion.
r=[1.,2.,3.];v=[.1,.2,.3];done=[False,True,False];nv=.4;ga=Tensor([0.]);expected=[];refga=0.
for ri,vi,di in reversed(list(zip(r,v,done))):
    ga=(ri+.99*(not di)*nv-vi+.99*.95*(not di)*ga).realize()
    refga=ri+.99*(not di)*nv-vi+.99*.95*(not di)*refga
    np.testing.assert_allclose(ga.numpy(),[refga],atol=1e-5);nv=vi
print(json.dumps({'status':'PASS','device':'CL','checks':['Gaussian log probability','PPO clipped surrogate','GAE reset boundary']}))
