# Qualcomm FLARE 适配工程

本目录是我们的工程入口，内部沿用原FLARE目录习惯：

- `waypoint_passing_train.py` / `waypoint_passing_eval.py`：训练和评估。
- `envs/scenario1_rigid_env.py`：刚性吊挂航点环境。
- `envs/genesis_bridge.py`：Vulkan物理与OpenCL训练接口。
- `config/envs/` / `config/train/`：环境与PPO配置。
- `runners/ppo.py`：OpenCL PPO和检查点。
- `utils/`：配置加载、GPU算子、硬件和日志。
- `tests/`：数值、物理和日志验证。

从工程根目录启动：

```bash
scripts/run.sh flare_q6a/waypoint_passing_train.py -e s1_waypoint_passing_rigid -B 2048
# 带资源监控和源码快照（原命令继续有效）
scripts/run.sh scripts/long_training.py -e s1_waypoint_passing_rigid
```

本目录原有train.py、evaluate.py等同级Python文件只转发旧命令和导入；修改实际实现时请进入对应子目录。根目录上游Flare/Genesis/rsl_rl_reference保持固定版本；模型在assets/，训练记录在records/。部署说明、配置对应关系和已知限制见根目录README.md。迁移不改变PPO逻辑、参数或检查点格式。
