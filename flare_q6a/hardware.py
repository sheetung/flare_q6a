"""Reject CPU OpenCL and non-Qualcomm devices before training."""
import ctypes
import json
from tinygrad import Device
from tinygrad.runtime.autogen import opencl as cl

def require_adreno():
    if Device.DEFAULT!='CL':raise RuntimeError('Set DEV=CL; CPU training is forbidden')
    dev=Device['CL'];kind=ctypes.c_uint64();vendor=ctypes.c_uint32()
    for field,out in [(cl.CL_DEVICE_TYPE,kind),(cl.CL_DEVICE_VENDOR_ID,vendor)]:
        err=cl.clGetDeviceInfo(dev.cl_dev,field,ctypes.sizeof(out),ctypes.byref(out),None)
        if err:raise RuntimeError(f'OpenCL device query failed: {err}')
    if not kind.value & cl.CL_DEVICE_TYPE_GPU:raise RuntimeError('Selected OpenCL device is CPU/software')
    if vendor.value!=0x5143:raise RuntimeError(f'Expected Qualcomm GPU, got vendor {vendor.value:#x}')
    return {'name':dev.device_name,'vendor_id':hex(vendor.value),'driver':dev.driver_version,'backend':'CL'}

if __name__=='__main__':print(json.dumps(require_adreno()))

def configure_main_cpu():
    """Optional process-local host dispatch affinity; existing workers stay free."""
    import os
    from pathlib import Path
    mode=os.environ.get('RL_MAIN_CPU','none')
    allowed=sorted(os.sched_getaffinity(0))
    if mode=='none':return {'mode':mode,'allowed_cpus':allowed}
    def score(cpu):
        root=Path(f'/sys/devices/system/cpu/cpu{cpu}')
        def number(path):
            try:return int(path.read_text())
            except (OSError,ValueError):return 0
        return number(root/'cpu_capacity'),number(root/'cpufreq/cpuinfo_max_freq'),cpu
    if mode=='auto':
        ranked=sorted(allowed,key=score)
        cpu=ranked[-1]
        if score(cpu)[:2]==(0,0):return {'mode':'none','reason':'CPU capacity/frequency unavailable','allowed_cpus':allowed}
    else:cpu=int(mode)
    if cpu not in allowed:raise ValueError(f'RL_MAIN_CPU={cpu} is outside allowed CPUs {allowed}')
    os.sched_setaffinity(0,{cpu})
    return {'mode':mode,'main_cpu':cpu,'capacity_and_max_frequency':score(cpu)[:2],
            'original_allowed_cpus':allowed,'scope':'calling training thread; already initialized workers retain affinity'}
