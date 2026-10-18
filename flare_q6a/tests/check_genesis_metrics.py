import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

"""Original Genesis integration: exact reward decomposition and timeout logging."""
import json,time
import numpy as np
from tinygrad import Tensor
from flare_q6a.envs.surrogate_env import ROOT
from flare_q6a.envs.genesis_bridge import GenesisBridge
from flare_q6a.utils.metrics import ExperimentLogger
out=ROOT/'records/logging_native_check';out.mkdir(parents=True,exist_ok=True)
env=GenesisBridge(2);env.env.max_episode_length=2
logger=ExperimentLogger(out,2,.01,1)
started=time.perf_counter()
for _ in range(8):
    obs,r,d,_,_=env.step(Tensor([[0.,0.,0.,0.]]*2,device='CL'))
    info=env.last_step_info
    np.testing.assert_allclose(sum(info['reward_parts'].values()),r.numpy(),atol=1e-6)
    assert np.isfinite(info['drone_error_m']).all()
    assert np.isfinite(info['payload_error_m']).all()
    assert np.isfinite(info['swing_deg']).all()
    logger.observe(r,d,info)
elapsed=time.perf_counter()-started
row={'iteration':1,'samples':16,'seconds':elapsed,'value_loss':0.,'policy_loss':0.,'kl':0.,'entropy':0.,'learning_rate':.0003}
logger.finish_iteration(row,elapsed,0.,1.);logger.close()
assert row['completed_episodes_this_iteration']==4
assert row['timeouts_only']==4 and row['crashes']==0
assert row['mean_episode_length']==3
assert len(row['episode_reward_terms'])==6
print(json.dumps({'status':'PASS','checks':['original Genesis rewards unchanged by observation tap','six exact weighted reward terms','pre-reset geometry finite','timeout episodes counted once','completed-episode length 3 steps']}))
