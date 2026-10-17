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
