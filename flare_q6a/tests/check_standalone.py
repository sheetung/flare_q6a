"""GPU check that runtime never opens external FLARE or its reference snapshot.

Use `scripts/run.sh flare_q6a/tests/check_standalone.py flex` or pass the
normal waypoint evaluation arguments (including --checkpoint).
"""
import os,sys,runpy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
blocked=[ROOT/'Flare',ROOT/'flare_q6a/reference']
def audit(event,args):
 if event not in ('open','os.chdir'):return
 if not isinstance(args[0],(str,bytes,os.PathLike)):return
 path=Path(os.fsdecode(args[0])).resolve()
 if any(path.is_relative_to(p) for p in blocked):raise RuntimeError(f'External/reference FLARE access forbidden: {path}')
sys.addaudithook(audit)
if sys.argv[1]=='flex':
 from flare_q6a.envs.genesis_bridge import GenesisBridge
 from tinygrad import Tensor
 import numpy as np
 env=GenesisBridge(2,rigid=False)
 obs,reward,*_=env.step(Tensor.zeros(2,4))
 assert obs.shape==(2,26) and np.isfinite(obs.numpy()).all() and np.isfinite(reward.numpy()).all()
 print('PASS: flexible environment GPU reset/step without external or reference FLARE')
else:
 sys.argv=['flare_q6a/evaluate.py',*sys.argv[1:]]
 runpy.run_module('flare_q6a.waypoint_passing_eval',run_name='__main__')
 print('PASS: checkpoint evaluation without external or reference FLARE')
