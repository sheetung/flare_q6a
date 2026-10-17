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

不指定 `--epochs`、`--minibatches` 时，读取 FLARE 的 5 epochs、4 minibatches；默认 `--envs 8 --steps 32`，对应 minibatch=64；原版 100 步采样需单独验证当前驱动。每次 PPO update 使用一次打乱顺序，在多个 epoch 中复用，与 rsl_rl v1.0.2 相同；不能整除 minibatch 的尾部样本会按原版规则舍弃并记录。

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
scripts/run.sh scripts/long_training.py \
  --envs 8 --steps 32 --iterations 500
```

省略 `--output` 时自动创建 `records/run_<UTC时间戳>`，并打印完整目录；也可用 `--output` 指定目录。

训练日志会实时显示在启动终端，同时保存到 `training.log`。

监督器输出 `status.json`、`manifest.json`、`telemetry.jsonl` 和 `latest_telemetry.json`；训练输出 `training.log`、`progress.json`、指标及检查点。`status.json` 记录训练 PID 和完成/失败状态；`manifest.json` 记录 Git commit 与完整命令。

GPU 占用仅在内核提供 DRM engine-time 计数器时计算，否则为 null；GPU 频率单独记录，不能替代百分比。NPU 未用于训练，NPU 百分比为 null，不能伪记为已测 0%。CPU 百分比同时记录单核 100% 基准和整机容量基准。温度包含所有可读 thermal zone，最大值不一定是 GPU 温度。

使用 systemd 用户服务时，任务可在聊天结束后继续；用户注销或系统重启可能终止任务。检查点可按前述方法恢复。监督器进程不做自动重启，失败会保留 exit code 和训练日志。

查看某次后台长训练的简洁实时状态：

```bash
python3 scripts/training_status.py --output records/run_long_20261017_01 --watch
```

`--output` 改为实际运行目录；不加 `--watch` 则输出一次。后台 systemd 用户服务可用 `systemctl --user status <服务名>` 查看，用 `systemctl --user stop <服务名>` 停止；只保留已完成轮次的检查点，未完成的轮次不会保存。


当前 Q6A/Mesa 25.2.8 实测发现：8 环境 × 100 步、4 minibatches（每批 200）在首轮 PPO 更新触发 RustiCL GPU lockup。减至每批 64 的配置已完成多轮并保存检查点，但完整长训练仍在验证；不能把小测试 PASS 视为任意批次配置都稳定。默认采样规模已调整为 8×32。

监督器支持 `--stall-timeout`（秒，默认 600；0 禁用），根据训练阶段进度判断停滞。超时会转储 Python 栈并结束实验，保留失败状态和最后已完成轮次的检查点，不会自动重启或回退 CPU。

## FLARE 风格训练日志与 TensorBoard

`training.log` 每轮输出可读块：采样/学习耗时、FPS、value/surrogate loss、KL、学习率、动作噪声、最近 100 个完整回合的平均回报和长度、六项奖励分解、航点到达数、无人机/负载位置误差、绳摆角、终止原因、总步数和 ETA。

结构化指标同时保存到 `metrics.jsonl` 和 `metrics.json`，TensorBoard 写入运行目录的 `tensorboard/`：

```bash
.venv-gpu/bin/tensorboard --logdir records/run_long_20261017_03/tensorboard --host 127.0.0.1 --port 6006
```

日志定义：

- `Mean reward` 是最近 100 个**完整回合的原始奖励总和**，与 `Mean reward per step` 区分；不含 PPO 超时 value 补偿。
- 奖励分项包含原始 FLARE 权重与 dt；相加对应环境实际返回的奖励。完整回合分项是累计值，不是原版 extras 中按固定 15 秒除算的值。
- 航点到达计数严格来自原版 `_at_target()`，没有把重置或存活当成成功率。
- 位置误差、摆角和终止原因在 reset 前取样，避免被重置后的零值或初始位置污染。
- `Swing from vertical` 是绳与世界竖直向下的夹角；`rope/body angle` 是原版安全约束使用的绳/机体夹角。
- `Crash` 包括越界、触地和摆角等原版 failure 条件；多个终止原因可同时出现。`timeout-only` 不含同时 failure 的回合。
- 这些是带探索噪声的训练指标，不替代固定任务、确定性策略的独立评估。

日志测试：`scripts/run.sh flare_q6a/check_metrics.py`（仅 CPU 记账测试）；`scripts/run.sh flare_q6a/check_genesis_metrics.py`（使用原版 GPU 仿真，应在没有其他训练竞争 GPU 时运行）。

## 原版 FLARE 命令行兼容

高通入口支持原版 `-e/--exp_name`、`-v/--vis`、`-B/--num_envs`、`--max_iterations`；原有 `--envs`、`--iterations` 仍可使用。当前实验支持 `s1_waypoint_passing`。自动目录包含实验名和 UTC 时间戳。

```bash
scripts/run.sh scripts/long_training.py -e s1_waypoint_passing -B 8 --max_iterations 500
# 在图形桌面终端打开原版仿真窗口和目标标记：
scripts/run.sh scripts/long_training.py -e s1_waypoint_passing -B 8 --max_iterations 500 -v
```

请只运行一个训练任务。`-v` 需要图形显示环境，窗口接入原版 HoverEnv 的 show_viewer 和 visualize_target。高通桌面窗口的实际显示效果仍需运行验证；无显示环境会明确报错。为避免已发现的 GPU 锁死，默认保持 8 环境、32 步，而非原版 2048 环境；监督器默认 500 轮，直接 train.py 默认 3 轮。--max_iterations 与 --iterations 在续训时均表示追加轮数。上游 Flare 子模块保持原样；勿直接运行其中依赖 CUDA 的训练入口。

## Q6A GPU 吞吐实测（2026-10-17）

`RL_PPO_BATCH_REALIZE=1` 启用 PPO 梯度批量提交与统计值合并读取，减少独立调度和读取次数；默认仍关闭，可进行对照。该优化已通过 rsl_rl v1.0.2 数值、梯度、参数更新和续训一致性检查。短测中 16 环境、32 步、4 minibatch（每批 128）去掉首轮后约 10.03 样本/秒，原 8 环境约 6.00。平均进程 DRM GPU engine 占用约从 44% 提高到 63%。不同配置也改变采样规模，不能据此推断收敛更快。

```bash
RL_PPO_BATCH_REALIZE=1 scripts/run.sh scripts/long_training.py \
  -B 16 --steps 32 --max_iterations 500 --stall-timeout 180
```

这是短测候选配置，尚未验证长时稳定性和收敛。32 环境 × 32 步 / 4 minibatch（每批 256）实测触发内核 GPU lockup，已停止；此前每批 200 也失败，当前不要直接放大 minibatch。保留默认 8 环境，未自动开启长训练。

性能记录见 `records/gpu_benchmark_results.json`、`records/gpu_batch_parity.log`、`records/gpu_benchmark_kernel_error.txt`。可用 `python3 scripts/benchmark_report.py records/<运行目录>` 重新统计。DRM 计数器短窗口可能因延迟结算超过 100%，不能当成真实满载；报告保留原始异常，使用较长窗口和吞吐比较，不将百分比截断伪装成满载。阶段归属仅为监控采样时的进度，并非精确 GPU profiler。

## 单段刚性杆悬停（新增任务）

新环境 `rigid_hover` 使用单段刚性杆、无人机端自由转动、负载端固定，负载在固定点悬停且保持无人机正下方。输入为无人机位置/速度/姿态/角速度与负载位置/速度，共 19 维；原柔性绳航点环境保留。详细输入顺序、动作、奖励、运行和测试说明见 [刚性杆悬停说明](flare_q6a/README_rigid_hover.md)。

```bash
RL_PPO_BATCH_REALIZE=1 scripts/run.sh scripts/long_training.py \
  -e rigid_hover -B 8 --steps 32 --max_iterations 500
```
