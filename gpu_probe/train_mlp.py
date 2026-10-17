"""Adreno smoke test: FLARE-sized MLP training, not a flight policy."""
import json
import time
import numpy as np
from tinygrad import Tensor, Device, nn, Context
from tinygrad.nn.optim import Adam
from tinygrad.nn.state import get_parameters

Tensor.manual_seed(1)
assert Device.DEFAULT == 'CL', Device.DEFAULT
layers = [nn.Linear(26, 128), nn.Linear(128, 128), nn.Linear(128, 4)]
params = get_parameters(layers)
assert all(p.device.startswith('CL') for p in params)
opt = Adam(params, lr=0.003)
rng = np.random.default_rng(1)
x_np = rng.normal(size=(256, 26)).astype(np.float32)
y_np = np.tanh(x_np[:, :4]).astype(np.float32)
x, y = Tensor(x_np, device='CL'), Tensor(y_np, device='CL')
losses = []
start = time.perf_counter()
with Context(TRAINING=1):
    for i in range(40):
        opt.zero_grad()
        pred = layers[2](layers[1](layers[0](x).tanh()).tanh())
        loss = (pred-y).square().mean()
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
        if i in (0, 9, 39):
            print(json.dumps({'step': i+1, 'loss': losses[-1]}), flush=True)
assert np.isfinite(losses).all()
assert losses[-1] < losses[0] * 0.5, losses
print(json.dumps({'status':'PASS', 'device':Device.DEFAULT,
 'parameter_devices':sorted(set(p.device for p in params)),
 'initial_loss':losses[0], 'final_loss':losses[-1],
 'seconds_including_compilation':time.perf_counter()-start}), flush=True)
