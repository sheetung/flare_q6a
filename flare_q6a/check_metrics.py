"""CPU-only logging checks: episode boundaries, reward units, tap transparency, TB."""
import json,tempfile
from pathlib import Path
import numpy as np
from metrics import ExperimentLogger,FlareMetricsTap

class FakeEnv:
    def __init__(self):
        self.num_envs=2
        self.env_cfg={'termination_if_roll_greater_than':180,'termination_if_pitch_greater_than':180,
            'termination_if_x_greater_than':3,'termination_if_y_greater_than':3,'termination_if_z_greater_than':2,
            'termination_if_close_to_ground':.1,'termination_cable_angle_threshold':1.571}
        self.base_pos=np.array([[0.,0.,1.],[0,0,.05]]);self.payload_pos=self.base_pos-[0,0,.48]
        self.cable_body_angle=np.zeros(2);self.rel_pos=np.array([[.2,0,0],[1.,0,0]])
        self.rel_pos_payload=np.array([[.2,0,.48],[1,0,.48]]);self.base_euler=np.zeros((2,3))
        self.episode_length_buf=np.array([3,7]);self.max_episode_length=1500
        self.safety_violation_count=np.zeros(2);self.max_safety_violations=150
        self.crash_condition=np.array([False,True]);self.reward_scales={'target':.1,'crash':-.1}
        self.reward_functions={'target':lambda:np.array([1.,2.]),'crash':lambda:np.array([0.,1.])}
    def _at_target(self):return np.flatnonzero(np.linalg.norm(self.rel_pos,axis=1)<.5)
    def reset_idx(self,indices):self.base_pos[indices]=[0,0,1];self.cable_body_angle[indices]=0
    def step(self):
        self._at_target();self.reset_idx(np.array([1]));return sum(self.reward_functions[k]()*self.reward_scales[k] for k in self.reward_functions)
base=FakeEnv();expected=base.step();env=FakeEnv();tap=FlareMetricsTap(env);tap.begin_step();actual=env.step();info=tap.finish(actual,[False,True])
np.testing.assert_array_equal(actual,expected);np.testing.assert_array_equal(env.base_pos,base.base_pos)
np.testing.assert_allclose(sum(info['reward_parts'].values()),actual)
assert info['causes']['ground'].tolist()==[False,True] # measured before reset
assert info['waypoint_hits'].tolist()==[1,0]
assert info['crashes'].sum()==1
with tempfile.TemporaryDirectory() as folder:
    out=Path(folder);logger=ExperimentLogger(out,2,.01,10)
    for reward,done in [([1.,2.],[False,False]),([3.,4.],[True,False]),([-1.,5.],[False,True])]:
        logger.observe(reward,done,{'reward_parts':{'target':np.asarray(reward)}})
    row={'iteration':1,'samples':6,'seconds':3.,'value_loss':.2,'policy_loss':-.1,'kl':.01,'entropy':5.,'learning_rate':.001}
    logger.finish_iteration(row,1.,2.,.9);logger.close()
    assert row['mean_episode_reward']==7.5 and row['mean_episode_length']==2.5
    assert row['episode_reward_terms']['target']==7.5 and row['completed_episodes_this_iteration']==2
    np.testing.assert_allclose(row['mean_raw_reward'],14/6)
    assert row['fps']==2 and row['total_samples']==6
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    events=EventAccumulator(str(out/'tensorboard')).Reload()
    assert events.Scalars('Train/mean_reward')[0].value==7.5
    assert events.Scalars('Train/mean_episode_length')[0].value==2.5
    assert len((out/'metrics.jsonl').read_text().splitlines())==1
print(json.dumps({'status':'PASS','checks':['reward tap preserves outputs/state','pre-reset termination diagnostics','episode boundaries across steps','weighted reward decomposition','TensorBoard scalar values']}))
