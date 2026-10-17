"""Probe Genesis-maintained Taichi fork; no implicit CPU fallback accepted."""
import json
import gstaichi as ti
print(json.dumps({'gstaichi_version':str(ti.__version__),'has_vulkan':hasattr(ti,'vulkan')}),flush=True)
assert hasattr(ti,'vulkan'), 'This gstaichi build exposes no Vulkan backend'
ti.init(arch=ti.vulkan,enable_fallback=False)
x=ti.field(dtype=ti.f32,shape=16)
@ti.kernel
def compute():
    for i in x: x[i]=i*2.0
compute()
assert ti.lang.impl.current_cfg().arch==ti.vulkan
assert x.to_numpy().tolist()==[float(i*2) for i in range(16)]
print(json.dumps({'status':'PASS','backend':'Vulkan'}),flush=True)
