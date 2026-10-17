"""FLARE configuration stays upstream; only topology and parallelism differ."""
import hashlib,json
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1]
RIGID_TASK='s1_waypoint_passing_rigid'
FLEX_TASK='s1_waypoint_passing'
MODEL=ROOT/'assets/rigid_suspended_system/rigid_suspended_system.urdf'
DEFAULT_ENVS=2048  # fastest measured rigid-task profile after native GPU row lookup.
DEFAULT_FLEX_ENVS=4  # original flexible solver has not been profiled at this scale.
DEFAULT_ITERATIONS=800  # original waypoint_passing_train.py command-line default

def default_envs(task):return DEFAULT_ENVS if task==RIGID_TASK else DEFAULT_FLEX_ENVS

def train_config():return yaml.safe_load((ROOT/'Flare/config/train/s1_train_config.yaml').read_text())
def environment_config(rigid=True):
    cfg=yaml.safe_load((ROOT/'Flare/config/envs/s1_env_config.yaml').read_text())
    if rigid:cfg['env_cfg']['num_of_dofs']=3
    return cfg

def task_signature(rigid=True):
    cfg=environment_config(rigid)
    model=MODEL if rigid else ROOT/'Flare/assets/suspended_system/real_sus_exp.urdf'
    return {'model_sha256':hashlib.sha256(model.read_bytes()).hexdigest(),
            'environment_sha256':hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()}
