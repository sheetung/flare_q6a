"""Register the server S2 model and configuration with the Vulkan bridge."""
import copy
import hashlib
import json
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[2]

def server_s2_config():
    config=yaml.safe_load((ROOT/'flare_q6a/config/envs/server_s2_rigid.yaml').read_text())
    config['env_cfg']['urdf_file']='assets/server_s2/real_sus_exp_rigid.urdf'
    return config

def server_s2_signature():
    cfg=server_s2_config()
    asset=ROOT/'flare_q6a/assets/server_s2/real_sus_exp_rigid.urdf'
    files=[ROOT/'flare_q6a/envs/scenario2_server.py',ROOT/'flare_q6a/utils/server_s2_mixer.py']
    meshes=sorted(asset.parent.glob('*.obj'))
    return {'model_sha256':hashlib.sha256(asset.read_bytes()).hexdigest(),
            'environment_sha256':hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest(),
            'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files+meshes}}

def setup_server_s2():
    from flare_q6a.envs import genesis_bridge as bridge
    from flare_q6a.envs.scenario2_server import RigidHoverEnv
    # Capture non-terminal metrics on every step; preserve pre-reset snapshot
    # on steps that terminate at least one environment.
    class ServerMetricsTap(bridge.FlareMetricsTap):
        def finish(self, reward, done):
            if not self.snapshot:
                self.snapshot=self.capture()
            return super().finish(reward,done)
    bridge.FlareMetricsTap=ServerMetricsTap
    bridge.HoverEnv=RigidHoverEnv
    bridge.environment_config=lambda rigid:copy.deepcopy(server_s2_config())
    bridge.task_signature=lambda rigid:server_s2_signature()
