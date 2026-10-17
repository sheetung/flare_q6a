"""Exact GPU row lookup, fresh JIT inputs and scaling with source size."""
import json,time
import numpy as np
from tinygrad import Tensor,TinyJit,Device
from hardware import require_adreno
from gpu_ops import take_rows
hardware=require_adreno();rng=np.random.default_rng(31)
for shape in [(7,),(7,26),(37,2,3)]:
    data=rng.normal(size=shape).astype(np.float32)
    idx=np.array([0,shape[0]-1,2,2,1,0],np.int32)
    actual=take_rows(Tensor(data),Tensor(idx)).numpy()
    np.testing.assert_array_equal(actual.view(np.uint32),data[idx].view(np.uint32))
# Native loads preserve bit patterns, including negative zero and NaNs.
data=np.array([-0.,np.inf,-np.inf,np.nan,1.],np.float32);idx=np.array([3,0,4,2,1],np.int32)
np.testing.assert_array_equal(take_rows(Tensor(data),Tensor(idx)).numpy().view(np.uint32),data[idx].view(np.uint32))
runner=TinyJit(lambda x,i:take_rows(x,i).realize())
saved=None
for _ in range(6):
    data=rng.normal(size=(257,26)).astype(np.float32);idx=rng.integers(0,257,size=101,dtype=np.int32)
    result=runner(Tensor(data),Tensor(idx)).clone().realize()
    np.testing.assert_array_equal(result.numpy(),data[idx])
    if saved is None:saved=result;snapshot=result.numpy().copy()
    else:np.testing.assert_array_equal(saved.numpy(),snapshot)
measurements=[]
for rows in (1024,51200):
    data=rng.normal(size=(rows,26)).astype(np.float32);idx=rng.integers(0,rows,size=4096,dtype=np.int32)
    x,i=Tensor(data).realize(),Tensor(idx).realize()
    runner=TinyJit(lambda a,b:take_rows(a,b).realize())
    for _ in range(3):runner(x,i)
    Device['CL'].synchronize();start=time.perf_counter()
    for _ in range(20):result=runner(x,i)
    Device['CL'].synchronize();elapsed=time.perf_counter()-start
    np.testing.assert_array_equal(result.numpy(),data[idx])
    measurements.append({'source_rows':rows,'selected_rows':4096,'columns':26,'milliseconds_per_call':1000*elapsed/20})
print(json.dumps({'status':'PASS','hardware':hardware,'checks':['scalar/matrix/higher-rank row lookup','duplicate and boundary indices',
    'exact float32 bit preservation','fresh JIT input buffers','saved outputs unchanged','large source lookup'],
    'native_lookup_timings':measurements}))
