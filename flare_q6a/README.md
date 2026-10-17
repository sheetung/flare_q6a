# 高通 GPU 适配层

安装、训练与移植见[根目录 README](../README.md)。当前默认任务 `s1_waypoint_passing_rigid`，原FLARE航点目标/26维观测/奖励/控制/PPO配置保持，默认刚性2048并行、采样100步；柔性参考默认4。性能配置见根README。刚性模型为 `assets/rigid_suspended_system/rigid_suspended_system.urdf`。

- rigid_waypoint_env.py：原环境副本，仅更换模型和刚性关节重置。
- genesis_bridge.py：GsTaichi Vulkan与CPU staging兼容层，原柔性任务仍可用-e s1_waypoint_passing。
- task_config.py：读取原上游配置、定义低并行默认值和模型/环境签名。
- ppo.py：rsl_rl v1.0.2的OpenCL移植，按逻辑minibatch累积/裁剪/KL/Adam，梯度/优化器/推理JIT，许可证见LICENSE.rsl_rl。
- gpu_ops.py：原生GPU行读取，替换tinygrad one-hot全扫描，不向rollout输入传播梯度。
- train.py / evaluate.py：训练、冻结策略评估与记录。
- check_waypoint_alignment.py：原逻辑对齐、质量、物理与部分重置验证。
- check_parity.py / check_microbatch.py：实际rsl_rl CPU参考与GPU PPO/Adam/断点续训对照。
- check_gpu_ops.py / check_rollout_jit.py：精确索引、JIT参数/轨迹缓冲与采样/价值推理验证。

自定义悬停环境及配置已删除，历史结果保留。长期稳定性与收敛尚未验证。
