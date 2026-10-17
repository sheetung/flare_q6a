"""CPU-only checks for episode quotas, aggregation and censored trajectories."""
import json
import numpy as np
from evaluation_metrics import EvaluationMetrics
m=EvaluationMetrics(2,.01,3)
def info(error,crash):return dict(payload_error_m=np.array(error),swing_deg=np.array(error)*10,
 hover_success=np.array([1.,0.]),crashes=np.array(crash),causes={'tilt':np.array(crash)})
m.observe([1,3],[True,False],[False,False],info([1,3],[True,False]),np.zeros((2,4)))
m.observe([2,4],[True,True],[False,True],info([2,4],[True,False]),np.zeros((2,4)))
r=m.summary();assert r['completed_episodes']==3 and r['requested_episodes_complete']
assert r['failures']==2 and r['safe_full_duration_episodes']==1 and r['evaluated_samples']==4
assert r['mean_payload_error_m']==2.5
assert r['mean_hover_success']==.5
assert abs(r['mean_episode_seconds']-.04/3)<1e-9
m.observe([99,99],[True,True],[True,True],info([99,99],[False,False]),np.ones((2,4))*2)
assert m.summary()['completed_episodes']==3 and m.summary()['evaluated_samples']==4
partial=EvaluationMetrics(2,.01)
partial.observe([1,2],[False,True],[False,False],info([9,3],[False,True]),np.zeros((2,4)))
r=partial.summary();assert r['completed_episodes']==1 and len(r['partial_episodes'])==1
assert r['mean_payload_error_m']==3 and r['partial_episodes'][0]['censored']
print(json.dumps(dict(status='PASS',checks=['exact total episode quotas','finished environments excluded','step-weighted completed episode statistics','timeout/failure distinction','censored episodes kept separate'])))
