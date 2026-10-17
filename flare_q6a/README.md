# GPU 适配层

完整安装、训练、续训和其他高通板迁移步骤见 [根目录 README](../README.md)。

- `genesis_bridge.py`：原版 FLARE/Genesis Vulkan + GsTaichi 兼容层。GPU 物理，CPU 观测/奖励与数据 staging；Vulkan ICD 可通过 `RL_VULKAN_ICD` 配置。
- `ppo.py`：与 rsl_rl v1.0.2 对齐的直接 sigma 参数、minibatch、Gaussian KL 自适应学习率、clipped PPO/value loss、梯度裁剪、Adam 和 schema-2 训练检查点。
- `train.py`：原版仿真采样与 GPU PPO；默认原版 rsl 超时 bootstrap，可显式选 terminal 变体。
- `evaluate.py`：加载训练检查点或仅模型文件，在仿真中评估，兼容早期 log-std 模型。
- `check_parity.py`：真实上游 CPU PPO 对照 GPU loss、梯度、参数更新、学习率及断点恢复。
- `check_ppo.py`：Gaussian/PPO/GAE 基础数值参考。
- `check_env.py`、`env.py`：独立简化环境诊断，非原版柔性绳物理。

检查点恢复优化器和随机数状态，但环境从 fresh reset 开始；尚无完整柔性绳物理状态快照。旧版仅网络参数的 NPZ 不能用于恢复优化器。

上游代码仍位于独立 submodule；不在此目录复制。PPO 移植许可证见 LICENSE.rsl_rl。长期稳定性、收敛、PX4/NPU 和第二块板的迁移均未验证。
