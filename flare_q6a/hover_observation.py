"""Shared 19-D simulator/deployment observation packing; quaternion order xyzw."""
import numpy as np

def rotation(q):
    q=np.asarray(q);q=q/np.linalg.norm(q,axis=-1,keepdims=True)
    w,x,y,z=np.moveaxis(q,-1,0)
    return np.stack([1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),
        2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),
        2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)],-1).reshape(*q.shape[:-1],3,3)

def pack_hover(drone_position,drone_velocity,orientation_xyzw,angular_velocity_flu,
               payload_position,payload_velocity,target_enu,rod_length=.48):
    """19 floats: drone error, velocity, xyzw, gyro, payload error, velocity."""
    q=np.asarray(orientation_xyzw,dtype=np.float32)
    norms=np.linalg.norm(q,axis=-1,keepdims=True)
    if np.any(norms<1e-6):raise ValueError('invalid orientation')
    q=q/norms;q=np.where(q[...,-1:]<0,-q,q)
    drone_target=np.asarray(target_enu)+np.array([0.,0.,rod_length])
    values=[np.asarray(drone_position)-drone_target,np.asarray(drone_velocity)/2.,q,
            np.asarray(angular_velocity_flu)/5.,np.asarray(payload_position)-np.asarray(target_enu),
            np.asarray(payload_velocity)/2.]
    out=np.concatenate(values,axis=-1).astype(np.float32)
    if not np.isfinite(out).all():raise ValueError('non-finite hover observation')
    return out
