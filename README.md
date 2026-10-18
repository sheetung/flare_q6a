# FLARE on Qualcomm：原柔性绳航点跟踪基线

当前默认任务恢复为原 FLARE Scenario 1 **柔性绳航点跟踪** `s1_waypoint_passing`：原模型、63内部自由度、混控/PID、观测、动作、奖励、目标与PPO参数保持原样。先比较高通GPU后端适配效果；默认仅将并行数从2048降低为512，每环境100步、5 epochs、4 minibatches、800轮不变。刚性任务作为显式选项 `-e s1_waypoint_passing_rigid` 保留，不作为这次原版基线。

已验证硬件平台：Radxa Dragon Q6A / Ubuntu 24.04 ARM64 / Python 3.12 / Adreno 643。Vulkan物理和OpenCL PPO在GPU运行；原FLARE观测、奖励、PID/mixer和跨框架数据搬运使用CPU。NPU未接入，代码不连接PX4或发送实飞指令。其他高通板必须重新验证真实GPU驱动与数值/物理测试。

## 工程目录（沿用 FLARE 的组织习惯）

```text
rl_radxa/
├── .venv-gpu/                 # Python虚拟环境
├── Flare/                     # 可选原仓库，本机保留但运行不再依赖
├── Genesis/                   # 固定版本物理引擎
├── rsl_rl_reference/          # 原PPO数值对照
├── flare_q6a/                 # 我们的高通适配工程入口
│   ├── waypoint_passing_train.py
│   ├── waypoint_passing_eval.py
│   ├── envs/
│   │   ├── scenario1_env.py       # 原柔性绳航点环境（默认）
│   │   ├── scenario1_rigid_env.py # 刚性吊挂航点环境
│   │   ├── genesis_bridge.py     # Vulkan物理/OpenCL接口
│   │   └── surrogate_env.py      # 实验替代环境
│   ├── config/
│   │   ├── envs/s1_env_config.yaml
│   │   └── train/s1_train_config.yaml
│   ├── runners/ppo.py         # OpenCL Actor/Critic与PPO
│   ├── utils/                 # 配置、GPU算子、硬件和日志
│   ├── tests/                 # 物理、数值和独立运行验证
│   ├── assets/                # 刚性/柔性模型及本地网格
│   ├── reference/Flare/       # 原工程源码、配置、资源和文档快照
│   └── train.py / evaluate.py # 旧入口兼容转发
├── scripts/                   # 部署、启动和资源监控工具
└── records/                   # 训练输出、验证结果和修改记录
```

适配代码、模型和网格集中在 `flare_q6a/`，训练和部署不需要根 `Flare/`。`envs/scenario1_env.py`、`utils/mixer.py` 和柔性资源从固定上游复制；未用到的源码、配置、文档和图片也保存在 `reference/Flare/` 供对照，来源与文件SHA256见 `reference/SOURCE.json`（不复制原训练日志、检查点、Git元数据和缓存）。参考目录不参与运行。`Genesis/` 和 `rsl_rl_reference/` 仍为子模块。

两份运行YAML位于 `flare_q6a/config/`，初始内容与固定上游一致；刚性任务仅在加载后将内部自由度设为3。根目录 `scripts/`、`records/` 及旧Python入口继续兼容已有命令。刚性URDF只改变网格相对路径，模型签名规范化这段路径以兼容旧checkpoint，物理参数不变。

直接训练（入口命名与原FLARE一致）：

```bash
scripts/run.sh flare_q6a/waypoint_passing_train.py -e s1_waypoint_passing -B 512
```

需要资源监控、源码快照和终端日志时，继续使用 `scripts/run.sh scripts/long_training.py -e s1_waypoint_passing`。两者调用同一训练实现。

## 原柔性基线的启动方式

```bash
scripts/run.sh scripts/long_training.py
# 等价的显式任务选择
scripts/run.sh scripts/long_training.py -e s1_waypoint_passing -B 512
```

默认带原策略随机采样噪声，观测和动作均沿用原版，不新增扰动或奖励。512并行是当前柔性基线的实测硬件规模选择，尚未验证柔性2048并行；下面的2048性能数据仅来自刚性任务，不能当作柔性结果。B512对应逻辑minibatch12800样本；不同并行数对应不同逻辑minibatch样本数，比较学习曲线时需同时看总样本数。

## 刚性可选任务与原 FLARE 的对应关系

| 项目 | 原版 | 刚性可选任务 |
|---|---|---|
| 并行环境 | 2048 | **2048（刚性实测档）** |
| 每环境采样步数 | 100 | **100，读取原YAML** |
| epochs / minibatches | 5 / 4 | 相同 |
| 每个逻辑 minibatch 样本数 | 51200 | 51200 |
| GPU计算块 | 原CUDA实现 | 最多4096，整批统一裁剪/KL/Adam |
| 观测 / 动作 | 26 / 4 | 相同 |
| Actor/Critic | [128,128] Tanh | 相同 |
| 学习率、KL、gamma、lambda、噪声 | 原YAML | 相同 |
| CLI默认训练轮数 | 800 | 800；YAML runner.max_iterations为500，但原脚本learn使用CLI轮数 |
| 模型 | 31组柔性段、63内部DOF | 单根刚性杆、根部3转动DOF，负载端固定 |
| 保存间隔 | 100轮及正常结束 | 相同 |
| 初始回合年龄 | 随机分散 | 相同 |

目标仍是**无人机航点**：随机采样 x/y ∈[-1.5,1.5]m、z∈[0.5,1.5]m，距离<0.5m切换下一目标。没有另设负载目标。26维观测包含当前/下一航点误差、无人机速度、姿态旋转矩阵、动作历史与吊挂角度/角速度，完全沿用原任务。

奖励使用原六项和原权重：target=10、smooth=-0.0001、yaw=0.01、angular=-0.0002、crash=-10、cable_angle_safety=0.03；全部按原逻辑乘dt=0.01，失败实际扣0.1，没有悬停版的-1/-5惩罚。原角速度PID、混控、RPM换算、100Hz及内部2子步保持不变，不套用已删除悬停环境的控制器或逐子步推力补偿。

高通后端仍需GsTaichi兼容层与tinygrad PPO移植，不能在本机直接使用原CUDA训练器。原PPO移植已做rsl_rl v1.0.2数值对照，含原版51200样本逻辑minibatch。GPU行索引采用原生间接读取，避免tinygrad默认one-hot全扫描；参数/梯度缓冲独立物化，配合JIT复用执行图。刚性环境 `flare_q6a/envs/scenario1_rigid_env.py` 从固定上游源码派生；AST检查确保除构造和重置外全部方法一致。刚性环境的部分重置采用完整qpos写入和全连续ID运动学刷新，绕过已确认的旧Genesis部分ID刷新缺陷。上游子模块不改写。

## 固定上游版本

| 上游 | 提交 |
|---|---|
| [FLARE](https://github.com/BEI11HAI/Flare) | 41f03b94f2f724205e5e24617010467aae2a3f98 |
| [Genesis](https://github.com/Genesis-Embodied-AI/Genesis) | 382cf4ca12c0c142adcf2fa7675eef65caf0c661 |
| [rsl_rl v1.0.2](https://github.com/leggedrobotics/rsl_rl) | 2ad79cf0caa85b91721abfe358105f869a784121 |

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

## 4. 验证

```bash
scripts/run.sh flare_q6a/utils/hardware.py
scripts/run.sh flare_q6a/tests/check_waypoint_alignment.py
scripts/run.sh flare_q6a/tests/check_waypoint_alignment.py --physics
scripts/run.sh flare_q6a/tests/check_parity.py
scripts/run.sh flare_q6a/tests/check_gpu_ops.py
scripts/run.sh flare_q6a/tests/check_microbatch.py --samples 1600 --microbatch-size 400 --epochs 5
scripts/run.sh flare_q6a/tests/check_rollout_jit.py --envs 1025 --microbatch-size 4096
```

对齐检查涵盖原函数AST、环境配置、质量与拓扑；物理检查涵盖26维观测、奖励分解、刚性长度、部分环境重置、目标切换和超时。PPO参考检查使用CPU PyTorch作数值参考，待测PPO仍在OpenCL GPU上执行。测试PASS不代表收敛。

## 5. 训练

```bash
cd /home/radxa/Documents/rl_radxa
scripts/run.sh scripts/long_training.py -e s1_waypoint_passing
```

默认柔性任务512并行、800轮、100步采样、5epochs/4minibatches；目录自动带实验名/时间戳，训练日志实时显示并保存。图形桌面需要窗口时加 `-v` 并调小-B；大规模性能测试均为headless，窗口显示仍需实机验证。需要500轮可加 `--max_iterations 500`。

采样保留原版100步。柔性模型有63内部DOF，当前默认512并行；刚性大规模测试结果不能直接推广到柔性。早期B4限制属于旧执行图/索引路径的故障边界，不是Adreno的并行硬件上限。新适配支持大逻辑minibatch：计算块自动取 `min(逻辑minibatch,4096)`，完整minibatch累积后才统一裁剪梯度、按完整KL调节学习率、执行一次Adam；仍然每轮5×4=20次更新。`-B 16` 对应400样本整批，已通过20次更新的原版数值对照。可用 `--microbatch-size` 显式覆盖计算块，不等同于修改 `--minibatches`。不同并行数每轮样本量不同，轮数不能直接比较收敛。

短测试：

```bash
scripts/run.sh scripts/long_training.py -e s1_waypoint_passing -B 512 --max_iterations 2
```

训练高斯探索噪声保持开启。`scripts/run.sh` 默认启用下列已通过数值检查的执行优化；它们不修改奖励、PPO目标或更新次数：

| 环境变量 | 默认 | 用途 |
|---|---|---|
| RL_PPO_JIT | 1 | 复用梯度计算执行图，参数显式输入 |
| RL_PPO_JIT_ADAM | 1 | 复用裁剪/Adam图，状态和学习率显式输入 |
| RL_ROLLOUT_JIT | 1 | 复用策略推理图，噪声显式输入，保留独立轨迹缓冲 |
| RL_PPO_BATCH_REALIZE | 1 | 合并GPU提交 |
| RL_MAIN_CPU | auto | 已初始化工作线程保持原亲和，主提交线程选择允许CPU中容量/频率最高核 |

前四项可设为0作对照，`RL_MAIN_CPU=none` 关闭线程绑定；不修改系统governor、频率或超频。`RL_TORCH_THREADS` 可选限制CPU数据准备线程，默认不改PyTorch设置；它不控制GPU反向传播。推理块上限1024，训练块上限4096。原生GPU行索引始终启用，不依赖CPU高级索引。直接调用 `flare_q6a/waypoint_passing_train.py` 时 --steps/--epochs/--minibatches 支持显式覆盖；修改这些超参数后与原PPO配置存在差异。

柔性基线任务 `-e s1_waypoint_passing` 默认512并行，尚未验证63自由度柔性模型的大规模配置。

checkpoint.npz按原版每100轮及正常训练结束保存；若提前Ctrl+C，未保存的更新会丢失。metrics.json、metrics.jsonl、training.log、TensorBoard记录训练指标；manifest.json记录命令、提交、执行开关与源码SHA256，source_snapshot/保存该次代码/模型/config；config.json及checkpoint记录原配置、新模型/任务签名与实际执行设置。status=PASS只表示流程校验通过。

## 6. 续训与评估

```bash
scripts/run.sh scripts/long_training.py -e s1_waypoint_passing \
  --resume records/<运行目录>/checkpoint.npz --max_iterations 100
scripts/run.sh flare_q6a/waypoint_passing_eval.py -e s1_waypoint_passing \
  --checkpoint records/<运行目录>/checkpoint.npz --envs 4 --episodes 4 --mode sample
```

续训轮数表示追加轮数；恢复模型、Adam与随机数状态，物理状态fresh reset，不是原轨迹精确重放。模型/环境签名及PPO配置不符会拒绝恢复；删除的19维悬停策略不能用于本任务。

评估默认sample带噪声；显式--mode mean关闭探索作对照，权重不更新。自动保存冻结checkpoint、SHA256、种子、逐回合长度/回报/误差/摆角/终止原因与汇总。安全飞满回合不等于完成航点跟踪，重点看航点到达与误差。

## 7. 日志、记录与移植

```bash
python3 scripts/training_status.py --output records/<运行目录> --watch
tail -f records/<运行目录>/training.log
```

GPU占用来自本进程DRM engine计数差分，短窗口可因驱动结算延迟出现异常，频率不是占用率；NPU未使用。每次修改的原因、行为和验证结果写入 records/progress.md 并提交本地Git。历史悬停日志/测试结果保留作审计，不是当前任务配置。

其他高通板需要Linux ARM64/Python3.12/glibc≥2.38，以及真实Adreno Vulkan/OpenCL。ICD路径不同可设置 `RL_VULKAN_ICD=/绝对路径/vendor_icd.json`。驱动、轮子、计算块/并行性能、长时稳定性和收敛需重新测试；本工程尚未验证PX4实飞、NPU或第二块开发板。无需CPU训练回退，也不会自动推送远程。

## 8. 本机性能验证（2026-10-17）

刚性原FLARE任务，100步采样、5个epoch、4个逻辑minibatch，GPU训练块最多4096；以下速度排除首次编译轮，按总样本/总耗时计算。

| 并行环境 | 连续轮数 | 稳定阶段样本/秒 |
| --- | --- | --- |
| 512 | 3 | 2363 |
| 1024 | 3 | 3370 |
| 2048 | 3 | 3774 |
| 默认2048入口 | 10 | 3737 |

默认入口共完成2,048,000样本、200次Adam更新，进程峰值RSS约1.014GiB，系统可用内存最低4.516GiB，传感器最高74.3°C。B16从第10轮追加到第11轮也通过，原checkpoint未改变。详见 [性能报告](records/hardware_tuning_results.json)、[连续验证](records/native_default_stability_check.json)、[B16续训](records/native_b16_resume_check.json)。

核心优化是原生GPU行索引替代tinygrad高级索引的全源扫描，以及梯度、Adam和推理JIT；保留原PPO逻辑批次、全局裁剪、KL调度和更新次数。旧索引大批次曾发生GPU lockup，其高占用不算稳定结果。默认选择以有效吞吐为准，不保证占用恒定100%。

实际原版PPO数值对照、非整除微块及强制梯度裁剪、断点状态恢复、跨1024推理块和bootstrap检查均通过。冻结策略加载评估仅2个0.03秒测试回合，权重不变；只验证执行链路。10轮验证不代表长期稳定、收敛或实飞通过，NPU尚未使用。

原 `Flare/` 已被根Git忽略，现可缺省；部署脚本不再下载它。使用 `scripts/run.sh flare_q6a/tests/check_standalone.py flex` 检查独立柔性环境；同一脚本也接受正常评估参数，审计钩子禁止读取根Flare及reference目录。训练源码快照包含本地实际模块和所有模型/网格资源。

刚性对照任务仍可用 `scripts/run.sh scripts/long_training.py -e s1_waypoint_passing_rigid` 启动。柔性默认环境为 `flare_q6a/envs/scenario1_env.py`，柔性模型为 `flare_q6a/assets/suspended_system/real_sus_exp.urdf`，网格与材质均在同目录，运行无需根Flare。

## 柔性并行实测（2026-10-18）

同原柔性模型、100步/5epochs/4minibatches、GPU执行开关，两个档位各连续3轮/60次Adam更新；速度排除首轮编译后按后两轮总样本/总耗时计算：

| 柔性并行数 | 样本/秒 | 结果 |
| --- | --- | --- |
| 128 | 346.8 | 3轮通过 |
| **512（当前默认）** | **1117.4** | 3轮通过 |

B512比B128吞吐约3.2倍；默认B4是此前未测柔性扩容的临时保守选择，现已替换。B512逻辑minibatch12800，GPU计算块最多4096，依旧每轮20次Adam更新。柔性63DOF显著增加共享内存开销，进程RSS不能覆盖全部GPU内存；不能直接套用刚性B2048。尚未测试柔性2048或硬件最大容量，-B仍可显式覆盖。详见[柔性性能报告](records/flexible_hardware_tuning_results.json)与[内存/数值验证](records/flexible_parallel_validation.json)。三轮仅验证流程和短程吞吐，不证明长期稳定或收敛。
