"""Linear-cost GPU row lookup for detached PPO rollout storage.

Pinned tinygrad 0.14 advanced Tensor indexing uses one-hot masking and a
reduction over every source row. This kernel performs one indexed load per
output element and participates in tinygrad scheduling/JIT capture.
"""
from tinygrad import Tensor,dtypes
from tinygrad.uop.ops import UOp,KernelInfo

def _take_rows_kernel(out:UOp,source:UOp,index:UOp):
    width=source.numel()//source.shape[0]
    i=UOp.range(out.numel(),0)
    row=index.flatten().index(i//width).load().cast(dtypes.weakint)
    value=source.flatten().index(row*width+i%width).load()
    return out.flatten().index(i).store(value).end(i).sink(arg=KernelInfo(name='ppo_take_rows',opts_to_apply=()))

def take_rows(source:Tensor,index:Tensor):
    """Select nonnegative in-bounds row indices; no gradient to rollout data."""
    if not source.device.startswith('CL') or index.device!=source.device:raise ValueError('take_rows requires same OpenCL device')
    if index.ndim!=1 or index.dtype!=dtypes.int32:raise ValueError('take_rows expects one-dimensional int32 indices')
    if source.ndim<1 or source.shape[0]<1:raise ValueError('empty source rows')
    shape=(index.shape[0],*source.shape[1:])
    out=Tensor.empty(*shape,device=source.device,dtype=source.dtype).is_param_(False)
    return out.custom_kernel(source.detach().contiguous(),index.detach().contiguous(),fxn=_take_rows_kernel)[0].detach()
