# FLARE on Qualcomm：吊挂无人机强化学习

在高通开发板上运行公开 FLARE 的 Genesis 柔性绳仿真，用 Adreno GPU 执行 PPO 神经网络训练。

**当前已验证平台：Radxa Dragon Q6A，Ubuntu 24.04，ARM64，Python 3.12，Adreno 643。** 其他高通板需先通过硬件后端检查；SoC 有 Adreno/Hexagon 不代表操作系统已提供兼容驱动。

本工程仍处于适配和验证阶段，没有可直接实飞的训练策略。代码不会连接 PX4 或发送飞行指令。

## 计算分工

| 部分 | 实际后端 |
|---|---|
| 原版 Genesis 刚体与柔性绳物理 | GsTaichi Vulkan / Adreno |
| 原版 FLARE 观测、奖励、PID/mixer | PyTorch CPU 张量 |
| 跨框架数据搬运 | CPU staging |
| 策略、价值网络、采样、GAE、反向传播、Adam | tinygrad OpenCL / Adreno |
| Hexagon NPU | 尚未接入 |

这是 GPU 物理与 GPU 训练结合的方案，尚非全 GPU 零拷贝。日志中的 PyTorch CPU staging 警告不表示 PPO 在 CPU 上训练。

## 仓库结构与上游版本

```text
Flare/                 上游 submodule，原始任务、配置、URDF
Genesis/               上游 submodule，指定历史版本物理引擎
rsl_rl_reference/      上游 submodule，PPO 数值对照，不用于实际训练
flare_q6a/             GPU PPO、Genesis 兼容桥、评估、测试
scripts/               安装和运行入口
requirements-q6a.lock  本地隔离环境包版本，不含主机 ROS 包
records/               执行记录与测试结果；模型文件不提交 Git
```

| 上游 | 固定提交 |
|---|---|
| [FLARE](https://github.com/BEI11HAI/Flare) | `41f03b94f2f724205e5e24617010467aae2a3f98` |
| [Genesis](https://github.com/Genesis-Embodied-AI/Genesis) | `382cf4ca12c0c142adcf2fa7675eef65caf0c661` |
| [rsl_rl v1.0.2](https://github.com/leggedrobotics/rsl_rl) | `2ad79cf0caa85b91721abfe358105f869a784121` |

上游源码通过 submodule 固定，不复制到主仓库。更新上游需重新验证兼容性；不要直接切换到最新 Genesis。

## 1. 获取源码

当前只维护本地 Git，没有配置或推送远程。以后有远程地址时：

```bash
git clone --recurse-submodules <本工程仓库地址> rl_radxa
cd rl_radxa
```

已有主仓库但缺少上游代码时：

```bash
git submodule update --init --recursive
```

## 2. 准备系统和 GPU 驱动

以下安装命令针对 Q6A 的 Ubuntu 24.04：

```bash
sudo apt-get update
sudo apt-get install -y python3.12-venv python3.12-dev build-essential git \
  mesa-opencl-icd clinfo vulkan-tools libgl1 libegl1 libglfw3
RUSTICL_ENABLE=freedreno clinfo -l
vulkaninfo --summary
```

OpenCL 必须列出 GPU，例如 `rusticl / FD643`；Vulkan 必须有真实 Adreno 设备。仅出现 llvmpipe 不合格。

系统依赖安装参考：[瑞莎 OpenCL 教程](https://docs.radxa.com/dragon/q6a/system-config/opencl)。其他开发板应使用板厂对应驱动，不能直接套用 Q6A 的 Mesa 安装方法。

## 3. 安装隔离环境

```bash
scripts/setup.sh
```

安装脚本检查 Linux ARM64/Python 3.12/glibc ≥ 2.38，清除继承的 ROS PYTHONPATH，创建 `.venv-gpu`，安装锁定依赖并编译指定 Genesis 的网格扩展。需要网络、数 GB 可用磁盘和编译工具。

原版 Taichi 没有本工程所需的官方 Linux ARM64 wheel，因此使用 GsTaichi 4.6.0。`genesis_bridge.py` 提供显式导入/内部类型兼容层，并替换旧版私有张量导出优化。Genesis 以 `--no-deps` 安装；`pip check` 的 `genesis-world requires taichi` 是此替代方案的已知元数据差异，不能据此宣称完全无依赖警告。

## 4. 验证后再训练

```bash
scripts/run.sh flare_q6a/hardware.py
scripts/run.sh flare_q6a/check_ppo.py
scripts/run.sh flare_q6a/check_parity.py
scripts/run.sh flare_q6a/genesis_bridge.py
```

`check_parity.py` 使用真实 rsl_rl v1.0.2 在 CPU 上作数值参考，实际待测 PPO 在 GPU 上运行。它检查 loss、梯度、minibatch 更新、KL 学习率和优化器/RNG 恢复。该 CPU 用途只是测试。

可选简化模型诊断：

```bash
scripts/run.sh flare_q6a/check_env.py
```

简化模型不是原始 FLARE 柔性绳模型，不应用作实飞策略训练依据。

## 5. 启动原版 FLARE 短训练

```bash
scripts/run.sh flare_q6a/train.py --physics genesis \
  --envs 2 --steps 4 --iterations 2 --epochs 2 --minibatches 2 \
  --output records/start_run
```

这是小规模验证，不是收敛训练配置。通过后逐步增加环境数和 rollout 长度；不要直接照搬桌面 NVIDIA 上的 2048 环境配置。

不指定 `--epochs`、`--minibatches` 时，读取 FLARE 的 5 epochs、4 minibatches；默认 `--steps 100`。每次 PPO update 使用一次打乱顺序，在多个 epoch 中复用，与 rsl_rl v1.0.2 相同；不能整除 minibatch 的尾部样本会按原版规则舍弃并记录。

默认 `--timeout-bootstrap rsl` 匹配 v1.0.2 的当前状态 value 超时补偿。`--timeout-bootstrap terminal` 使用捕获的终止观测 value，是显式算法变体，不应与原版结果混为一谈。

输出：

- `metrics.json`：每轮 loss、KL、学习率、更新数、重置数、耗时。
- `checkpoint.npz`：模型、Adam、学习率、迭代数及随机数状态。
- `model.npz`：仅策略/价值网络参数，用于评估。
- `result.json`：运行配置与阶段结果；`PASS` 表示流程通过，不表示策略收敛。

## 6. 断点续训与评估

恢复时使用与原运行相同的 PPO 配置：

```bash
scripts/run.sh flare_q6a/train.py --physics genesis \
  --envs 2 --steps 4 --epochs 2 --minibatches 2 --iterations 2 \
  --resume records/start_run/checkpoint.npz --output records/resumed_run
scripts/run.sh flare_q6a/evaluate.py --physics genesis --envs 2 --steps 100 \
  --checkpoint records/resumed_run/model.npz
```

`--iterations` 表示追加的训练轮数。恢复网络、优化器与 RNG，**仿真从新 reset 开始**；当前不保存完整柔性绳物理状态，因此不是连续轨迹的精确重放。PPO 配置不匹配会拒绝恢复。早期只有网络参数的模型不能当作训练检查点；旧 log-std 模型可用于兼容评估。

## 7. 迁移到其他高通开发板

| 条件 | Q6A | 其他高通板 |
|---|---|---|
| Linux ARM64 + Python 3.12 | 已测 | 需验证 wheel 与 glibc |
| Adreno Vulkan compute | Mesa Turnip 已测 | 需板厂驱动支持 |
| Adreno OpenCL 编译/执行 | RustiCL Freedreno 已测 | 需排除 CPU OpenCL 和缺失算子 |
| GsTaichi 4.6.0 + 指定 Genesis | 已短测 | 必须重跑检查 |
| Hexagon NPU | 未使用 | 本工程没有 NPU 部署支持 |

Vulkan ICD 路径不同可配置：

```bash
export RL_VULKAN_ICD=/你的硬件驱动路径/vendor_icd.json
scripts/run.sh flare_q6a/genesis_bridge.py
```

默认仅加载 Q6A 的 `/usr/share/vulkan/icd.d/freedreno_icd.json`。不要改用软件 Vulkan 来绕过错误。专有 OpenCL 驱动可不需要 RustiCL 设置，但仍需确认 tinygrad 选中真实 GPU。本工程尚未在第二块开发板上验证，锁文件是 Q6A 基线，不是所有高通板的兼容承诺。

## Git 与实验管理

源码、脚本、锁文件、上游 commit 和文本记录提交 Git。虚拟环境、缓存、凭据、训练模型与大体积运行目录排除。

```bash
git status
git log --oneline
git submodule status
```

模型应独立备份，并记录对应主仓库 commit、运行配置与文件校验值。当前没有远程同步；以后配置远程不会自动上传被忽略的模型文件。

## 当前边界

目前尚未验证长时间训练稳定性、收敛和论文指标。下一步是扩大 rollout、固定评估任务、多随机种子训练和资源监控，然后才是 PX4/动捕接口、内环辨识与 sim-to-real。适配层包含 rsl_rl 算法移植，其许可证见 `flare_q6a/LICENSE.rsl_rl`；各 submodule 遵循各自上游条款，本仓库不替上游授予额外权限。

详细历史见 [records/progress.md](records/progress.md)。

## 长训练与资源监控

后台监督器可运行固定轮数的训练，并每 5 秒记录资源数据：

```bash
scripts/run.sh scripts/long_training.py --output records/run_long_example \
  --envs 8 --steps 100 --iterations 200
```

监督器输出 `status.json`、`manifest.json`、`telemetry.jsonl` 和 `latest_telemetry.json`；训练输出 `training.log`、`progress.json`、指标及检查点。`status.json` 记录训练 PID 和完成/失败状态；`manifest.json` 记录 Git commit 与完整命令。

GPU 占用仅在内核提供 DRM engine-time 计数器时计算，否则为 null；GPU 频率单独记录，不能替代百分比。NPU 未用于训练，NPU 百分比为 null，不能伪记为已测 0%。CPU 百分比同时记录单核 100% 基准和整机容量基准。温度包含所有可读 thermal zone，最大值不一定是 GPU 温度。

使用 systemd 用户服务时，任务可在聊天结束后继续；用户注销或系统重启可能终止任务。检查点可按前述方法恢复。监督器进程不做自动重启，失败会保留 exit code 和训练日志。

查看某次后台长训练的简洁实时状态：

```bash
python3 scripts/training_status.py --output records/run_long_20261017_01 --watch
```

`--output` 改为实际运行目录；不加 `--watch` 则输出一次。后台 systemd 用户服务可用 `systemctl --user status <服务名>` 查看，用 `systemctl --user stop <服务名>` 停止；只保留已完成轮次的检查点，未完成的轮次不会保存。
