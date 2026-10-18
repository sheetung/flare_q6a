# Qualcomm FLARE 适配工程

本目录是我们的工程入口，内部沿用原FLARE目录习惯：

- `waypoint_passing_train.py` / `waypoint_passing_eval.py`：训练和评估。
- `envs/scenario1_rigid_env.py`：刚性吊挂航点环境。
- `envs/genesis_bridge.py`：Vulkan物理与OpenCL训练接口。
- `config/envs/` / `config/train/`：环境与PPO配置。
- `runners/ppo.py`：OpenCL PPO和检查点。
- `utils/`：配置加载、GPU算子、硬件和日志。
- `tests/`：数值、物理、日志和禁止外部依赖访问的验证。
- `assets/`：刚性、柔性模型和全部网格。
- `reference/Flare/`：原FLARE源码、配置、资源和文档快照，供阅读，不参与运行。

从工程根目录启动：

```bash
scripts/run.sh flare_q6a/waypoint_passing_train.py -e s1_waypoint_passing -B 512
# 带资源监控和源码快照（原命令继续有效）
scripts/run.sh scripts/long_training.py
```

本目录原有train.py、evaluate.py等同级Python文件只转发旧命令和导入；修改实际实现时请进入对应子目录。运行不需要根Flare/，实际环境与混控均已复制并改为本地导入。Genesis/rsl_rl_reference仍为固定子模块；模型在本目录assets/，训练记录在根records/。来源及SHA256见reference/SOURCE.json。部署说明、配置对应关系和已知限制见根目录README.md。迁移不改变PPO逻辑、参数或检查点格式。

当前默认使用原柔性绳任务做硬件适配基线，原模型/奖励/目标/PID/mixer/PPO保持原样，仅降低并行到512。刚性任务需显式-e s1_waypoint_passing_rigid；其2048性能报告不是柔性性能。
