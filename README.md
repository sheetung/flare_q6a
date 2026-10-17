# FLARE on Qualcomm：刚性吊挂航点跟踪

当前任务是原 FLARE Scenario 1 航点跟踪，替换为独立的单段刚性吊挂模型，降低并行环境数；自定义 rigid_hover 悬停环境及其19维输入、奖励配置已经删除，历史记录保留在 Git 与 records 中。

已验证硬件平台：Radxa Dragon Q6A / Ubuntu 24.04 ARM64 / Python 3.12 / Adreno 643。Vulkan物理和OpenCL PPO在GPU运行；原FLARE观测、奖励、PID/mixer和跨框架数据搬运使用CPU。NPU未接入，代码不连接PX4或发送实飞指令。其他高通板必须重新验证真实GPU驱动与数值/物理测试。

## 与原 FLARE 的对应关系

| 项目 | 原版 | 当前默认 |
|---|---|---|
| 并行环境 | 2048 | **4** |
| 每环境采样步数 | 100 | **100，读取原YAML** |
| epochs / minibatches | 5 / 4 | 相同 |
| 每个 minibatch 样本数 | 51200 | 100 |
| 观测 / 动作 | 26 / 4 | 相同 |
| Actor/Critic | [128,128] Tanh | 相同 |
| 学习率、KL、gamma、lambda、噪声 | 原YAML | 相同 |
| CLI默认训练轮数 | 800 | 800；YAML runner.max_iterations为500，但原脚本learn使用CLI轮数 |
| 模型 | 31组柔性段、63内部DOF | 单根刚性杆、根部3转动DOF，负载端固定 |
| 保存间隔 | 100轮及正常结束 | 相同 |
| 初始回合年龄 | 随机分散 | 相同 |

目标仍是**无人机航点**：随机采样 x/y ∈[-1.5,1.5]m、z∈[0.5,1.5]m，距离<0.5m切换下一目标。没有另设负载目标。26维观测包含当前/下一航点误差、无人机速度、姿态旋转矩阵、动作历史与吊挂角度/角速度，完全沿用原任务。

奖励使用原六项和原权重：target=10、smooth=-0.0001、yaw=0.01、angular=-0.0002、crash=-10、cable_angle_safety=0.03；全部按原逻辑乘dt=0.01，失败实际扣0.1，没有悬停版的-1/-5惩罚。原角速度PID、混控、RPM换算、100Hz及内部2子步保持不变，不套用已删除悬停环境的控制器或逐子步推力补偿。

高通后端仍需GsTaichi兼容层与tinygrad PPO移植，不能在本机直接使用原CUDA训练器。原PPO移植已做rsl_rl v1.0.2数值对照。刚性环境 `rigid_waypoint_env.py` 从固定上游源码派生；AST检查确保除构造和重置外全部方法一致。部分重置采用完整qpos写入和全连续ID运动学刷新，绕过已确认的旧Genesis部分ID刷新缺陷。上游子模块不改写。

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
scripts/run.sh flare_q6a/hardware.py
scripts/run.sh flare_q6a/check_waypoint_alignment.py
scripts/run.sh flare_q6a/check_waypoint_alignment.py --physics
scripts/run.sh flare_q6a/check_parity.py
```

对齐检查涵盖原函数AST、环境配置、质量与拓扑；物理检查涵盖26维观测、奖励分解、刚性长度、部分环境重置、目标切换和超时。PPO参考检查使用CPU PyTorch作数值参考，待测PPO仍在OpenCL GPU上执行。测试PASS不代表收敛。

## 5. 训练

```bash
cd /home/radxa/Documents/rl_radxa
scripts/run.sh scripts/long_training.py -e s1_waypoint_passing_rigid -B 4
```

默认800轮、100步采样、5epochs/4minibatches；目录自动带实验名/时间戳，训练日志实时显示并保存。图形桌面需要窗口时加 `-v`，窗口显示仍需实机验证。需要500轮可加 `--max_iterations 500`。

**不要沿用旧的 -B 16 --steps 32 命令**。本板已出现minibatch200/256/400的GPU锁死；保留100步和4minibatches时默认4环境，对应100样本/批。当前入口在场景构建前拒绝minibatch≥200，避免重复触发已知故障；低于200不代表均已验证。增大-B需先测试。并行降低512倍意味着相同轮数的样本量也降低512倍，不能承诺与桌面原版同轮数收敛。

短测试：

```bash
scripts/run.sh scripts/long_training.py -e s1_waypoint_passing_rigid -B 4 --max_iterations 2
```

训练探索噪声仍开启。`RL_PPO_BATCH_REALIZE=1` 是可选的GPU提交优化，不改变PPO公式或奖励，默认关闭。直接调用 `flare_q6a/train.py` 时 --steps/--epochs/--minibatches 支持显式覆盖，覆盖后不再属于仅改变模型/并行数的对齐配置。

checkpoint.npz按原版每100轮及正常训练结束保存；若提前Ctrl+C，未保存的更新会丢失。metrics.json、metrics.jsonl、training.log、TensorBoard记录训练指标；manifest.json记录命令和源码提交，config.json记录原配置与新模型/任务签名。status=PASS只表示流程校验通过。

## 6. 续训与评估

```bash
scripts/run.sh scripts/long_training.py -e s1_waypoint_passing_rigid -B 4 \
  --resume records/<运行目录>/checkpoint.npz --max_iterations 100
scripts/run.sh flare_q6a/evaluate.py -e s1_waypoint_passing_rigid \
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

其他高通板需要Linux ARM64/Python3.12/glibc≥2.38，以及真实Adreno Vulkan/OpenCL。ICD路径不同可设置 `RL_VULKAN_ICD=/绝对路径/vendor_icd.json`。驱动、轮子、锁死边界、长时稳定性和收敛需重新测试；本工程尚未验证PX4实飞、NPU或第二块开发板。无需CPU训练回退，也不会自动推送远程。
