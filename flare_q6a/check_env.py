"""Physical invariants and deterministic one-step reference, no flight hardware."""
import json
import numpy as np
from tinygrad import Tensor, Device
from env import CableEnv

assert Device.DEFAULT=='CL'
e=CableEnv(4)
# Hover of drone and suspended payload requires total system weight.
hover=2*(e.mass+e.load_mass)/(3.5*e.mass)-1
act=Tensor([[hover,0.,0.,0.]]*4)
for _ in range(100):
    obs,r,d,_,_=e.step(act)
assert not d.numpy().any()
np.testing.assert_allclose(e.p.numpy(),np.tile([0.,0.,1.],(4,1)),atol=2e-5)
np.testing.assert_allclose(e.v.numpy(),0,atol=2e-5)
np.testing.assert_allclose(e.s.numpy(),np.tile([0.,0.,-1.],(4,1)),atol=2e-5)
# Initially vertical cable and zero rotation: both masses accelerate together.
e=CableEnv(4);action=np.array([[.2,0,0,0]]*4,np.float32)
obs,r,d,_,_=e.step(Tensor(action))
thrust=.6*3.5*e.mass*9.81
az=thrust/(e.mass+e.load_mass)-9.81
np.testing.assert_allclose(e.v.numpy()[:,2],az*.01,rtol=1e-4,atol=1e-6)
np.testing.assert_allclose(e.p.numpy()[:,2],1+az*.0001,rtol=1e-5)
np.testing.assert_allclose(e.s.numpy().sum(1),-1,atol=1e-5)
assert obs.shape==(4,26)
# Force a crash and ensure independent auto-reset produces actual reset observation.
e.p=Tensor([[0.,0.,-.2]]+[[0.,0.,1.]]*3)
obs,r,d,_,_=e.step(Tensor([[hover,0,0,0]]*4))
assert d.numpy()[0]
np.testing.assert_allclose(e.p.numpy()[0],[0,0,1],atol=1e-5)
assert np.isfinite(obs.numpy()).all()
print(json.dumps({'status':'PASS','device':Device.DEFAULT,'checks':['100-step hover equilibrium','vertical acceleration reference','26D observations','crash auto-reset']}))
