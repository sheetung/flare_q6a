# Q6A FLARE GPU 适配记录

日期：2026-10-17（UTC）

## 目标与约束

使用公开 FLARE；仿真采样自主学习；GPU 执行训练，GPU/NPU 执行部署推理。不以 CPU 训练作为替代。实机为 PX4，动捕提供无人机与负载状态。凭据不写入记录。

## 源码基线

- Flare：41f03b94f2f724205e5e24617010467aae2a3f98，目录 ../Flare。
- Genesis：382cf4ca12c0c142adcf2fa7675eef65caf0c661，目录 ../Genesis。
- FLARE README 指定 rsl_rl v1.0.2，尚未安装。
- PPO；26 维观测、4 维动作；Actor/Critic 隐层 [128,128]、Tanh；默认 2048 个环境。
- 动作是油门及三轴角速度，仿真使用自带 PID/mixer 转电机 RPM。接 PX4 前需匹配内环和推力模型。

## 后端检查

- Vulkan：真实集成 GPU Turnip Adreno 643，Mesa 25.2.8；另有 CPU llvmpipe，不能误选。
- Genesis 指定版本有 Vulkan/Taichi 物理后端，但 utils/misc.py:get_device 在没有 CUDA/Intel XPU 时将 PyTorch 张量放在 CPU。因此 Vulkan 物理仿真不等于 GPU PPO 训练。
- FLARE 训练器硬编码 cuda:0，环境及 mixer 默认 CUDA，不能只改设备字符串。
- 原始 clinfo 显示 0 平台；mesa-opencl-icd 未安装。

## 已执行的环境配置

1. sudo apt-get install -y mesa-opencl-icd：成功。安装依赖并更新部分 Mesa/LLVM 配套包。
2. RUSTICL_ENABLE=freedreno clinfo -l：成功，rusticl / FD643。
3. sudo apt-get install -y python3.12-venv：成功，同时升级软件源要求的 Python 3.12 配套包。
4. python3 -m venv .venv-gpu：成功。
5. .venv-gpu/bin/python -m pip install tinygrad numpy：成功；锁定版本见 gpu_requirements.txt。

## GPU 最小训练验证

脚本：../gpu_probe/train_mlp.py。使用与 FLARE actor 相同的 26→128→128→4 MLP、Tanh、Adam；固定合成监督目标。这只验证训练算子，不是 PPO 或吊挂训练。

运行：

```bash
RUSTICL_ENABLE=freedreno DEV=CL .venv-gpu/bin/python gpu_probe/train_mlp.py > records/gpu_mlp_probe.log 2>&1
```

脚本强制 CL，断言所有参数在 CL 上；检查 loss 有限且下降。NumPy 只生成输入与读取结果。初次使用旧版 Tensor.train API 失败，已改为 tinygrad 0.14.0 的 Context(TRAINING=1)。当前运行结果见 gpu_mlp_probe.log。

实测结果：PASS。40 次前向/反向/Adam 更新，参数设备 CL；loss 从 0.3870134 降至 0.01274393；含首次编译共 13.51 秒。说明本机 Adreno OpenCL 可以完成该小型网络的训练更新，不代表 PPO 已跑通或已取得加速收益。

## 待完成

- GPU 与参考实现的梯度/更新数值对照及稳态吞吐量验证。
- Genesis Vulkan 最小物理场景及 ARM64 依赖验证。
- 决定保留 Genesis 并交换张量，或移植 FLARE 动力学到 GPU 张量环境。尚未改写原版仿真。
- PPO 采样与更新、原版奖励/观测一致性、端到端吞吐量。
- QNN/NPU 推理未验证，实机接口未接入。

资料：https://docs.radxa.com/dragon/q6a/system-config/opencl

## FLARE 适配阶段 1：2026-10-17

### Genesis ARM64 依赖阻碍

在隔离环境执行 `pip install taichi torch`，解析阶段失败：`No matching distribution found for taichi`。此次联合安装没有完成 torch 安装。核对 Taichi 1.7.4 官方 PyPI 发布文件：Linux wheel 为 x86_64，ARM64 wheel 为 macOS，未提供 Linux ARM64 wheel。此问题不能通过改变 FLARE 的 device 字符串解决。源码编译尚未尝试；不能判定 Genesis Vulkan 在本机不可用，也未实测原版物理性能。

### 已实现

独立目录 `../flare_q6a`，保持 `../Flare` 原版文件不变：

- `env.py`：tinygrad OpenCL 批量简化吊挂环境；读取 FLARE 原配置，26D 观测、4D 油门/角速度动作及奖励映射。
- `train.py`：OpenCL Actor/Critic、Gaussian 采样、GAE、PPO clipped surrogate、clipped value loss、entropy、梯度裁剪与 Adam。
- `check_env.py`：100 步悬停平衡、垂直加速度解析参考、26D 观测及单环境 crash 重置。
- `evaluate.py`：NPZ checkpoint 重新加载及 GPU 仿真评估。
- `README.md`：运行方法、模型假设及与原版 FLARE 差异。
- 安装 PyYAML 6.0.3；版本见 gpu_requirements.txt。

所有训练计算强制 CL，拒绝 CPU 训练回退。NumPy 仅用于结果读取、解析参考和模型 I/O。

### 实测

- 环境测试 PASS，见 env_checks.log。
- PPO 训练 PASS：8 环境 × 16 步 × 3 轮，2 次 full-batch 更新/轮；384 条样本，6 次更新；含编译、采样、更新和保存共 58.73 秒；首个参数张量最大变化 0.0018098354。见 ppo_smoke.log 和 ppo_smoke/result.json。
- 首次 checkpoint safetensors 保存因缺少 clang 失败，改为 NumPy NPZ 文件保存；失败产物重命名为 model.safetensors.failed，不应加载。
- 正常 checkpoint：ppo_smoke/model.npz。仅模型参数，没有优化器/环境/RNG 状态，不能视为精确续训快照。
- 评估结果见 ppo_eval.log。

### 实质限制

此阶段跑通的是 **FLARE 接口形状与奖励映射下的 GPU PPO 原型**，不是原版 Genesis FLARE 的物理等价复现。

简化环境：始终张紧无质量单绳、点质量负载、一阶角速度响应，未包含原版 63 DOF 柔性绳、原版 PID/mixer、碰撞、松绳或 PX4 动态。负张力仍可能发生，因此该模型不能用于松绳任务，策略不能实飞。

PPO：full-batch 固定学习率，尚未实现 rsl_rl minibatch 和自适应 KL。奖励在重置前计算，与公开原版顺序有差异。详细差异见 flare_q6a/README.md。

PASS 不代表训练收敛、论文复现、加速收益或实飞验证。下一阶段需决定 Genesis ARM64 源码构建或 GPU 柔性绳动力学移植，并做物理一致性与 PPO 数值对照；不应直接扩展当前简化环境做实飞训练。

评估实测 PASS：重新加载 NPZ，在 8 个 GPU 环境运行 32 步，状态有限，平均奖励 -0.00015823。该短评估未发生重置，重置行为由独立环境测试覆盖。

追加调查：发现 GsTaichi 4.6.0 提供 Linux ARM64 wheel。已开始安装并以 gpu_probe/genesis_backend_probe.py 检查 Vulkan 后端（禁用 fallback）。它不是指定版 Taichi 的已验证替代品。

### 原版物理后端追加验证

- GsTaichi 4.6.0 安装成功，`genesis_backend_probe.py` Vulkan 内核 PASS（禁止 fallback），见 genesis_backend_probe.log。
- 这推翻了“只能先停在缺失 ARM64 wheel”的阻碍，但尚未证明它与 Genesis 指定提交兼容。
- 新增 genesis_bridge.py：将旧 Taichi 命名空间映射到 GsTaichi，原版 Genesis Vulkan + 原版 FLARE 环境的桥接实验。尚在安装依赖和验证，不能视为可用。
- 桥接路径包含 CPU PyTorch staging 和原版 CPU 观测/奖励计算；训练仍使用 CL。不是全 GPU 环境路径。
- 新增 train.py --physics surrogate|genesis，默认 surrogate；genesis 模式需验证后使用。
- 最新 PyPI torch ARM64 解析到 CUDA 包，下载已停止；改用 torch 2.5.1 用于 API staging，避免无用 CUDA 依赖。
- Gaussian log probability、PPO clip、GAE reset 边界对照 NumPy PASS，见 ppo_numeric_checks.log。

### 原版 Genesis 构建与兼容修复

- ARM64 仿真依赖安装成功，见 genesis_deps_install.log；追加 pyvista/screeninfo 及依赖见 genesis_extra_deps.log。
- 指定提交 Genesis 0.2.1 的 Cython 网格扩展本机构建成功，见 genesis_build.log。
- GsTaichi DataType 已重命名为 DataTypeCxx；在桥接脚本中添加兼容别名。
- 首轮原版场景已构建并 reset，但读取角速度时，旧 Genesis 私有 launch 优化与 GsTaichi Kernel 参数布局不兼容（arguments → arg_metas）。桥接脚本已改为 GsTaichi 自带数据导出内核，继续测试。
- 此轮不修改原版 FLARE/Genesis Python 源码，只在桥接进程中设置兼容层。

## 阶段 2：原版 FLARE 物理 + GPU PPO 已打通

此节为当前最新结论，取代阶段 1“只能运行简化环境”的状态。

### 通过的实际测试

1. 原版 FLARE 场景构建、reset 和 4 次 step：PASS（genesis_flare_probe.log）。
2. 进一步强制 Vulkan ICD 为 `/usr/share/vulkan/icd.d/freedreno_icd.json`，排除 llvmpipe。
3. 原版 Genesis/FLARE + OpenCL PPO：2 环境 × 4 步 × 2 轮，2 次 full-batch 更新/轮。16 条样本、4 次更新，首个参数张量最大变化 0.0012032688；训练及保存阶段 17.37 秒，**不含前面的 Genesis 初始化/场景构建时间**。PASS，见 ppo_genesis.log 和 ppo_genesis/result.json。
4. checkpoint ppo_genesis/model.npz 已保存。
5. 加载该 checkpoint，2 个原版环境运行 8 步；测试专用回合长度设为 2 步，触发 4 次超时和 4 次 reset，终止状态及返回状态均有限：PASS，见 genesis_eval.log。

### 最新代码路径

- 默认 `train.py --physics genesis`，不再默认简化环境。
- 省略 output 创建新的时间戳目录。
- genesis_bridge.py 在 reset 前捕获终止观测，修复原版接口不返回 terminal observation 导致的 timeout bootstrap 问题。
- 原版柔性绳物理走 Vulkan Adreno；PPO 更新走 OpenCL Adreno。
- 原版观测/奖励/mixer 和跨框架搬运仍在 CPU；不是全 GPU 零拷贝。不能把 CPU staging 警告误读为 CPU 物理或 CPU 神经网络训练。
- NPU 尚未使用。完整说明及运行命令见 ../flare_q6a/README.md。

### 尚未满足的目标

尚无收敛策略，未复现论文指标；没有长时间吞吐量/稳定性测试；PPO 尚未逐项对齐 rsl_rl minibatch、自适应 KL 和断点续训；未接 PX4/动捕或验证实飞。

## 阶段 3：PPO 对齐与本地 Git 工程化

用户要求：先使用本地 Git，不同步远程。已初始化 main 分支，Flare/Genesis/rsl_rl_reference 以固定 SHA 的 submodule 登记；没有配置主仓库 remote。

- 根 README 包含安装、验证、训练、实用续训、评估、Git 管理和其他高通板迁移条件。
- scripts/setup.sh、scripts/run.sh：隔离 ROS PYTHONPATH，固定 CL，安装锁定 ARM64/Python3.12 依赖。
- requirements-q6a.lock 从虚拟环境实际 site-packages 导出，排除主机 ROS 包和 Genesis editable 行；Genesis 单独构建。
- 凭据、虚拟环境、模型、缓存排除 Git；上游保留各自许可。PPO 移植包含 rsl_rl BSD 许可。

### PPO 原版差异修正

核对真实 rsl_rl v1.0.2 后发现：原版直接优化动作 sigma，而早期原型优化 log(sigma)。现已改为直接 sigma，并保留旧模型推理加载转换。

- minibatch：按 v1.0.2 一次 permutation，在多个 epoch 复用；非整除尾部按原规则丢弃并记录。
- KL：Gaussian 闭式公式与 +1e-5；按原版 ×1.5/÷1.5 调整学习率，范围 1e-5～1e-2。
- 优势标准差采用样本标准差 correction=1，与 torch 默认一致。
- 默认超时补偿改为原版当前 value；terminal value 为显式可选变体。
- schema-2 checkpoint 保存模型、Adam m/v、bias correction、学习率、更新数、迭代数、NumPy shuffle RNG、Python RNG、tinygrad GPU RNG，若已加载 torch 也保存其 RNG。
- 原版物理环境不快照，续训 fresh reset，不是完全相同的连续物理轨迹。

### 对照测试

check_parity.py 与真实上游 PPO CPU reference 使用相同权重、轨迹和 minibatch 顺序；待测计算在 CL GPU。

结果 PASS：loss、梯度、上游真实 PPO 多次 minibatch 参数更新、KL 学习率分支、Adam 续训、GPU RNG 和 shuffle RNG 恢复。

梯度最大绝对误差 1.1920928955e-7；4 次更新后的网络参数最大绝对误差 4.5560300350e-6（容差 5e-6，并非位级相同）。见 ppo_parity.log。

测试期间修正了参考测试的 critic privileged observation 设置，并用浮点容差替代学习率浮点精确相等；没有通过放宽参数更新容差掩盖算法错误。

基础 Gaussian/PPO/GAE 对照重新 PASS。长时间稳定性与收敛仍未验证。

### 原版实际训练与恢复复测

- 对齐后的 PPO 在原版 Genesis 环境训练 1 轮：8 样本、2 次 minibatch 更新，学习率由 0.0003 调整至 0.000675；保存 schema-2 checkpoint。见 ppo_aligned.log。
- 在新进程重建环境、加载 checkpoint 后追加 1 轮：迭代数从 1 到 2，更新数从 2 到 4，学习率继续至 0.00151875。PASS，见 ppo_resumed.log。此短测证明实际恢复路径可用，不证明长期训练收敛。
- 新增硬件检查：OpenCL 必须是 GPU，vendor ID 必须为 Qualcomm 0x5143；本机 FD643 检查 PASS，见 hardware_verified.log。防止迁移时误选 CPU OpenCL。
- 安装脚本增加 glibc >= 2.38 检查，匹配当前锁定 wheel 的最低要求。

### Git 验证

- 首次本地提交 b51206d，main 分支，使用现有 Git 提交身份；没有主仓库 remote。
- 在 /tmp 的独立克隆中，从本地上游副本取出全部 3 个固定 submodule；检查点、虚拟环境和凭据未进入 Git tracked files。
- shell 语法与 Python 编译检查通过。
- 此验证覆盖 Git 打包和本机已安装环境，不等于另一块高通板已部署成功，也没有在新系统重装全部依赖。

## 长训练启动准备

用户要求启动长训练并监控资源。选择首次持续运行配置：原版 Genesis，8 个环境，每轮 100 步，200 轮，原配置 5 epochs / 4 minibatches；共请求 160,000 条仿真样本和 4,000 次 minibatch 更新。不预先承诺时长或收敛。

新增 scripts/long_training.py：后台监督、每 5 秒资源采样、Git/命令 manifest、完成/失败状态；记录 GPU 频率、可用 DRM engine 利用率、进程 CPU/RSS、系统内存/swap、所有 thermal zone。没有可读占用计数器时用 null，不用频率冒充占用。NPU 未参与，记录未使用且占用未知。

train.py 增加 rollout/update/checkpoint 阶段进度文件与原始奖励均值；未改变 PPO 更新公式。语法检查通过。

### 已启动的长训练

- 启动时间：2026-10-17 04:08:32 UTC。
- 服务：rl-flare-long-20261017-01.service（systemd 用户服务）。
- 运行目录：records/run_long_20261017_01。
- 训练 PID：14288；监督器 PID：14285（以运行目录 status.json 为准）。
- 实验源码 commit：a0c3b64906fc506e8f3092860a335172cc7a569a。
- 已进入 rollout 和 PPO update 阶段；后台服务保持运行。
- 训练进程启动后，DRM fdinfo 实际暴露 msm drm-engine-gpu 纳秒计数器；监控按 drm-client-id 去重，再用相邻采样的时间差计算训练进程引擎占用。先前在空闲进程检查不到，不等于训练时没有计数器。
- 2026-10-17 04:10:18 UTC 的 5 秒采样窗口：GPU 进程引擎占用约 18.2%，瞬时频率 315 MHz，GPU 传感器最大值 52.9°C；CPU 进程 40.2%（单核 100% 基准），RSS 1.16 GiB，CPU 传感器最大值 70.4°C。这是初始编译/第一轮阶段的样本，不能当作稳定性能结论。
- NPU 未参与，未测百分比。
- 新增 scripts/training_status.py 提供简洁只读状态，可 --watch 持续显示。

### 长训练 01 的驱动停滞与重启准备

首轮更新没有完成。线程显示 RustiCL 队列 dma_fence_default_wait；内核 04:12:49～04:13:01 UTC 多次记录 `hangcheck detected gpu lockup rb 0`、offending task 为本次训练的 rusticl queue t。已停止 rl-flare-long-20261017-01.service，保留全部监控文件；没有 checkpoint，不能宣称此配置训练成功。

暂不能确定根因，较大 minibatch=200 的 GPU kernel 是待验证因素。下一次试用 power-of-two minibatch=64：8 环境 × 32 步 / 4 minibatches，500 轮，共 128,000 条样本；5 epochs 不变。此调整不是已证实修复。

增加 PPO forward/backward/optimizer 的批次进度、SIGUSR1 Python 栈转储，以及监督器 progress timeout。新实验设置 180 秒无进展则转储并终止，保留失败状态，防止把活进程误报为正常训练。

### 替代长训练 02 正常推进

服务 rl-flare-long-20261017-02.service 于 04:14:32 UTC 启动，源码 commit 0927d959997498068604c1917db89a20a26f7b01；8×32、500 轮、minibatch 64。

已完成前 2 轮（每轮 256 样本、20 次更新），进入第 3 轮；检查点已实读验证 iteration=2、updates=40、sigma 正值且有限。前两轮耗时约 60.91 和 45.42 秒，完整实验可能数小时，尚未完成。

04:17:29 UTC 的 5 秒采样窗口：GPU 进程引擎占用约 33.36%、812 MHz、RSS 约 1.15 GiB。原实验 01 服务已确认 inactive/dead，02 active/running。

已把默认训练采样调整到当前已推进的 8×32/minibatch64，并在 README 标明较大批次锁死问题；仍需长期验证。当前服务通过显式参数运行，默认值调整不改变它的配置。最新入口及文件位置见 records/active_training.md。

## FLARE 风格日志补齐

用户要求日志输出原版 FLARE 式有用指标。新增 metrics.py：

- 非侵入式观察 `_at_target`、reward 函数和 reset 前状态；不修改奖励、动作或目标切换。
- 输出最近 100 个完整回合的回报、长度和奖励分项；同时保留每步奖励分项，单位为原版权重×dt 后的实际值。
- 航点到达计数、无人机/负载误差、世界竖直摆角、原版绳/机体夹角、failure/timeout-only 和可重叠的终止原因。
- 原版风格每轮文本、FPS、collection/learning time、KL、noise std、ETA、TensorBoard 及 metrics.jsonl。
- CPU 记账测试 PASS：episode 边界、加权分解、reset 前原因、观察 tap 保持状态/输出不变、TensorBoard scalar 值。
- 日志 Mean episode reward 用完整回合累计值；明确区别于每步奖励，以及原版 extras 按固定 15 秒归一化的数值。
- 使用下一次检查点切换新版本；保留旧日志和优化器状态，仿真按已有实用续训语义 fresh reset。

日志集成检查 PASS：原版 Genesis 8 步、2 环境，强制 3 步超时回合，共 4 个完成回合；六项加权奖励之和等于环境返回值；位置/摆角为 reset 前有限值；完整回合长度统计为 3 步；无需修改 PPO 算法。见 genesis_metrics_checks.log。

旧长训练 02 在第 49 轮 / 980 次更新的完整检查点保存后停止。新长训练 03 将恢复该检查点，追加 451 轮至总第 500 轮；网络、Adam、学习率和迭代数保留，环境 fresh reset，日志窗口从新进程开始积累（旧日志没有完整回合信息，不能补造）。

新版长训练 03 已恢复并实测输出第 50～52 轮可读日志，已继续进入第 53 轮。第 52 轮已统计 7 个完整回合，平均回报约 -0.16519、平均长度约 54.14 步；该轮记录 rope/body-angle failure 2 次、航点到达 0 次。指标是实测值，未补造旧日志缺失的回合信息。

TensorBoard 实时文件验证 PASS：已读到 39 个 scalar tag，包括 value loss、KL、原始每步奖励、无人机/负载误差、摆角。检查使用项目虚拟环境（系统 Python 未安装 tensorboard）。见 live_tensorboard_check.json。

运行目录和服务入口已更新到 records/active_training.md。修改仅增加观察和日志，不修改任务奖励、动作、PPO 更新或终止规则。

2026-10-17：长训练监督器的 --output 改为可选，默认自动生成 UTC 时间戳目录；直接训练入口统一使用可读时间戳，两个入口均打印输出路径。保留显式路径和原有训练行为。

2026-10-17：修正长训练监督器只写文件不显示终端的问题；独立线程实时跟随 training.log 输出到启动终端，保留资源监控和日志持久化。正在运行的旧进程不受修改影响，下次启动生效。

2026-10-17：两个高通训练入口新增原版 -e、-v、-B、--max_iterations 参数兼容，保留已有参数；-v 传递至原版 HoverEnv viewer 与目标标记。保持高通安全默认值、GPU 后端和上游子模块。图形窗口尚未实机运行验证，避免与用户正在运行的训练争用 GPU。

2026-10-17 GPU 性能测试：8×32 旧训练去掉首轮吞吐 6.003 样本/秒；16×32 原调度两轮、第二轮 8.388；16×32 批量梯度提交/统计合并读取三轮，去掉首轮 10.031 样本/秒，平均活动阶段 DRM engine 约 63.14%。优化用 RL_PPO_BATCH_REALIZE=1 显式开启，默认保持原实现。原版 rsl_rl parity PASS，参数误差 4.556e-6。1 秒监控出现延迟计数 >100% 异常，不作为真实满载证据；已增加较长窗口汇总工具。

32×32 批量提交、minibatch 256 在第一轮第一 minibatch 优化器锁死；06:02:24 UTC 内核 hangcheck 明确归因 rusticl queue t，服务立即停止且进程退出。未继续放大批次，也未修改 GPU governor/系统频率。退回 16×32 做恢复复测，全部保留旧检查点和实验目录。测试期间代码是 fd6c767 基础上的本次工作区改动，优化开关在 manifest 中记录，最终代码与汇总将一起提交。

GPU 恢复复测完成 PASS：16×32 合并提交再次完成两轮/40 次更新，正常保存 checkpoint.npz 与 model.npz；无运行中的测试遗留。最终汇总见 gpu_benchmark_results.json。未启动长训练，未把 16 环境短测结果宣称为长期稳定或收敛。

2026-10-17：新增独立实验 rigid_hover，保留原柔性绳 s1_waypoint_passing。按用户最新输入定义改为 19 维：无人机目标位置误差、世界速度、xyzw 姿态、机体系角速度、负载目标位置误差与世界速度；无话题转换或实机控制代码。固定负载目标 [0,0,0.52]、无人机正上方0.48m。单根刚性杆、负载固定，三个共点转动关节集中于无人机端，近似球铰；虚拟铰链微小质量明确记录。

验证发现 Genesis 0.2.1 每个物理子步清空电机外力：一次施加RPM后推进两个内部子步导致平均推力减半。新环境改为每控制周期两次显式5ms物理步、每步重新施加同一RPM。原环境未修改。修正前 records/run_rigid_hover_smoke 标记 superseded，不作为有效动力学训练结果。修正后悬停20步高度保持，速度约1.9e-7m/s，杆长最大误差2.98e-8m。物理检查 PASS：拓扑、19维观测顺序/缩放、标称推力平衡、动作下长度约束、奖励分解、超时/失败分离与reset前终止观测。

最终配置8环境×32步完成2轮/40次PPO更新，权重变化0.0259223，检查点记录任务、19维观测定义与环境配置；见 rigid_hover_training_check.json。加载检查点评估PASS：2环境8步、强制3步超时、4次timeout/reset、观测有限；见 rigid_hover_evaluation_check.log。旧航点checkpoint在场景构建前被拒绝。共享日志回归 check_metrics PASS。新任务窗口未实测、长时稳定性/收敛与真实PX4内环标定尚未验证，没有启动长期训练。

2026-10-17：用户长训33轮出现完成回合仅2–3步，排查发现新增 rigid_hover 的部分环境重置与固定 Genesis 内核不兼容，而非单纯训练慢。固定悬停16环境100步零失败；随机动作前100步也正常。延长到300步首次在105步正常越界，重置后负载仍保持1.8m误差和非零速度，随后1334次失败中1309次一步回合。

根因：Genesis _kernel_forward_kinematics_links_geoms 的 ndarray for 循环使用了索引而非传入的环境ID；部分重置时新环境的分开 set_pos/set_quat/set_dofs_position 路径又会从陈旧 dofs_state.pos 重建根 qpos，并返回陈旧负载状态。改为所选环境完整 qpos 原子写入、清零广义速度，再全连续环境索引刷新 FK。未改上游子模块、奖励、探索噪声或内环增益，以保持因果对照。reset_version=2 阻止旧错误环境配置续训，建议从头训练。

回归PASS：16环境非连续ID [15]、奇数ID及[0,15]重置后无人机/负载位置和速度正确，未重置环境qpos/速度保持不变；见 rigid_hover_partial_reset_checks.log。同动作种子123、初始种子1的300步修复前后前104步最大指标差0，均105步首次失败；修复后32次失败、0次一步回合、完成回合平均117.75步。详见 rigid_hover_reset_comparison.json；完整原始轨迹保留在忽略的 records/run_reset_diagnosis/。没有启动新的长期训练，尚未验证新策略收敛。

2026-10-17：按用户要求将 rigid_hover 的失败惩罚从 -5.0 降到 -1.0（仍不乘 dt）；其他奖励、物理参数和终止条件保持不变。配置差异检查通过。用户将从头重新训练，已有运行进程不会热加载此修改；旧奖励配置检查点续训会被环境配置一致性检查拒绝。

2026-10-17：按用户要求加入可审阅的冻结策略均值评估，并要求后续每次修改做记录。训练高斯探索保持不变；evaluate.py 扩展 mean/sample 模式、可复现初始状态 seed、完整回合配额、逐回合/汇总指标、失败原因、未完成轨迹分离、checkpoint复制冻结及SHA256、评估前后权重一致性断言。新增纯CPU统计回归PASS（仅统计检查，不训练）：配额、已结束环境排除、步加权统计、超时/失败区分与censored轨迹。

主训练 records/run_rigid_hover_20261017_094009_240109 的用户进程已停止（并非本次主动停止）；最新完整checkpoint为第154轮，第155轮未完成。未自动恢复或启动长训练。开始用第154轮冻结checkpoint做16环境/16完整回合、mean模式、seed123的真实GPU仿真评估，原环境配置/奖励不变；结果另行记录。此次修改基于 b1557ad 工作区，后续代码与验证记录一起提交。

第154轮冻结策略GPU评估完成，两组均16个完整回合、seed123、相同checkpoint SHA256 db8159fb5fa085af9a3f4d86150d1ab1d559ea880e1c65c2604fbba97bc8a285，评估前后权重一致。mean：平均1.4325s、最大1.61s、负载误差0.67554m、摆角19.862°、悬停条件7.8098%、安全飞满15s为0/16，14次位置越界、2次触地，动作截断0%。sample：平均2.13375s、最大2.82s、误差0.72460m、摆角22.363°、达标3.1927%、安全飞满15s为0/16，12次越界、3次摆角、1次倾角，动作分量截断37.8295%。两组执行状态PASS仅代表评估校验通过，不是悬停成功。

结论：去除探索使位置/摆角稍改善，但没有形成稳定悬停，甚至持续时间更短；不能把训练失败仅归因于探索。没有更改训练探索、奖励或PPO超参数，也没有自动重启已停止的原训练。结果见 mean_policy_evaluation_154.json 和 policy_evaluation_comparison_154.json；完整逐回合与冻结checkpoint保留在 records/run_eval_mean_154 与 records/run_eval_sample_154。评估代码新增负数种子/调试超时参数校验，原FLARE评估分支同时设置Torch种子，默认初始化seed仍保持1；不改变现有训练行为。

2026-10-17：按用户“加回去”的要求，将评估默认模式从 mean 改为 sample，默认动作=网络均值+已学习标准差×高斯噪声。显式 --mode mean 仍可关闭噪声作对照。训练采样本来一直有噪声，此次不修改训练算法、标准差、奖励或其他参数。命令行默认值及两个分支检查通过，既有运行进程不热加载新默认值。

2026-10-17：按用户明确要求撤销自定义悬停任务。删除 rigid_hover 环境、19维观测打包、配置、诊断测试与说明；旧刚性模型目录删除，新独立模型 assets/rigid_suspended_system/rigid_suspended_system.urdf。历史日志/checkpoint及Git审计记录保留。新任务 s1_waypoint_passing_rigid 从原FLARE Scenario1派生，AST验证除__init__与reset_idx外所有方法一致；目标采样/切换、26维观测、六项奖励、原PID/mixer/RPM、终止规则、100Hz/内部2子步全部恢复原逻辑，不继续使用自定义悬停逐子步推力补偿或-1/-5惩罚。

读取原训练YAML：100步、5epochs/4minibatches、同学习率/KL/gamma/lambda/噪声；恢复原CLI默认800轮、随机初始回合年龄、每100轮及结束保存。仅默认并行2048降为4，使minibatch100，避免此前200/256锁死；本次将实际跑4×100训练验证。新刚性杆将原绳段总质量0.0062kg分配为杆0.00618kg及两个虚拟铰链各1e-5kg，系统总质量仍0.3742kg。三转动DOF集中根部、负载固定。刚性qpos原子重置与全索引FK修复保留；新checkpoint记录模型/环境签名，旧悬停或不同物理模型拒绝续训。

静态对齐、配置、质量/拓扑PASS；实际Vulkan的26维观测、奖励分解、杆长、非连续ID重置、目标切换、超时PASS；原日志统计回归PASS，评估统计改用航点指标后回归PASS。当前无原训练进程，未主动删除旧运行数据，未自动开启长训练。新4×100训练验证结果将另行追加。

原FLARE刚性任务最终默认4×100训练验证PASS：两轮、800样本、40次优化器更新，minibatch100无驱动锁死，权重变化0.0098507；见 rigid_waypoint_training_check.json，实验源码e841787。最终checkpoint记录新任务、26维、原环境配置与模型签名，原失败权重-10（实际每次-0.1）。加载评估PASS：2环境、4完整回合、sample带噪声，强制原版3步超时验证4次重置、权重不变；测试horizon0.03s已明确记录，不表示飞满15s或收敛。见 rigid_waypoint_evaluation_check.json。评估统计补充航点到达总数，保持原目标定义；统计回归PASS。旧19维悬停checkpoint在物理初始化前被拒绝。没有启动长期训练，当前没有训练/评估残留进程。

2026-10-17：用户B16×100/4=400在第一轮末触发sigma断言。内核证据确认14:19:36起反复GPU lockup，offending task为该训练的rusticl queue，而非据断言推定正常学习发散；原日志无sigma数值，无法区分NaN/负值。增加每次更新前KL/梯度范数有限检查、更新后sigma实际数值与位置检查，以及ppo_failure.json。一次B16复现第一优化器更新同样锁死，已停止服务。驱动具体错误kernel未定位，未通过clamp修改原版sigma数学。

当前Q6A入口在场景构建前拒绝minibatch≥200，明确记录已知200/256/400故障，低于200仍需验证，非硬件并行上限。任务、奖励、探索噪声、100步/5epochs/4minibatches默认不改。新增检查与rsl_rl原版数值对照PASS，参数最大误差4.556e-6；停止复现后OpenCL概率/裁剪/GAE恢复检查PASS；B16入口拒绝检查exit2。见b16_failure_diagnosis.json、b16_gpu_lockup_kernel.log及相关检查日志。B16大批次尚未修复，没有启动长训练。

2026-10-17：按用户榨干硬件性能要求实现GPU microbatch梯度累积：默认compute chunk100，逻辑minibatch400分4块，按样本数加权梯度与KL；完整minibatch后仅一次全局裁剪、一次KL学习率调节、一次Adam，保留原5epochs/4minibatches=20更新。不再拦截逻辑大批次，仅约束单个GPU块<200；不改变任务、奖励、噪声或rollout100。监督入口转发--microbatch-size，默认100。

新增check_microbatch与实际rsl_rl v1.0.2对照：1600样本/4minibatches/2epochs，micro100，参数误差7.749e-7，Adam一二阶状态与学习率/更新次数PASS，58.217s。JIT第一版未显式传入更新后的参数，数值对照失败已拒绝；第二版将参数作为JIT输入并复制复用的梯度输出，PASS误差4.841e-7、12.497s（包含测试初始化，非端到端训练）。RL_PPO_JIT=1与RL_PPO_BATCH_REALIZE=1目前显式启用测试；真实B16连续3轮运行中，测试开始时基于9383f6a工作区本次代码改动。默认JIT尚未开启，结果后续追加。

2026-10-17 性能适配第二阶段：B16×100 micro100/JIT梯度/合并提交连续3轮PASS、60次Adam更新，后两轮55.729/61.869 samples/s；B32相同执行选项连续3轮PASS、60更新，后两轮91.606/100.630 samples/s，检查点已保存。测试开始时均为e820381之前/之后本次工作区，全部源改动由Git提交追踪。新增裁剪+Adam TinyJit（所有参数、状态与动态学习率显式输入）与rollout策略TinyJit（噪声显式输入，输出独立复制保留历史），参数/Adam状态对照PASS4.841e-7，11.410s；64环境12次rollout缓存测试PASS误差0，覆盖观测/噪声变化、两次参数修改、历史轨迹不被覆盖。末块37样本加权对照PASS4.712e-7。B64真实仿真连续3轮测试中，目前所有JIT开关仍为测试显式设置。

2026-10-17 B64首次全缓存运行在第1次采样前报TinyJit duplicate inputs，并非GPU锁死。复现确认全新参数的actor/critic部分同形随机参数共享底层Buffer；原CPU对照逐个numpy读取使该路径未暴露。修正为初始化及加载时给每个模型参数/优化器状态独立clone并逐个realize；新增train.py相同首权重snapshot顺序的fresh-init rollout回归PASS。原B64失败目录保留；run_microbatch_b64_v2_check已完成采样进入真实更新。新增性能汇总脚本，首轮排除，记录执行开关、吞吐、DRM进程engine、RSS/温度和checkpoint有限性，避免把不同优化组合误当仅并行数对照。

2026-10-17 继续适配：完整JIT checkpoint/Adam/GPU噪声与shuffle RNG续训数值回归PASS，参数误差5.856e-6。micro128（含128×3+16末块）实际rsl_rl对照PASS4.973e-7。B64全JIT连续3轮PASS，后两轮综合194.527 samples/s；B128 micro128+主提交线程自动选择cpu7（capacity1024/2.7072GHz，既有工作线程亲和不变）连续3轮PASS，后两轮约310.8 samples/s。只改本进程亲和，不修改系统频率或governor。默认微块改为auto：逻辑minibatch能被128整除用128，其余100；原PPO逻辑不变。

减少每microbatch统计的host同步，在GPU按样本权重累积5项统计、每逻辑minibatch读取一次；实际rsl_rl参数/Adam/KL对照PASS4.973e-7。采样和bootstrap推理也分块≤128，噪声仍在外部产生，不改变探索；257环境（两整块加1末块）对独立CPU PyTorch公式验证PASS，最大误差1.907e-6，覆盖更新权重、轨迹缓冲独立、随机层初始化与参数/优化器buffer独立。B256真实3轮测试中。监督器新增每次运行代码/模型/config只读快照与SHA256，避免工作区后续编辑影响实验审计；不复制密码或环境凭据。

2026-10-17 大块图适配复测：为界定旧故障是容量还是执行图问题，用内部实验开关及45/60s超时分别短测micro256/512/1024/2048/4096，全部通过实际rsl_rl参数、Adam状态、KL与更新数对照；最大参数误差依次3.475e-5、2.027e-6、2.384e-7、2.318e-6、5.350e-6，数值检查另增加绝对误差5e-5上限。过去故障的原B16逻辑400整批，以5epochs/20次Adam直接计算亦PASS4.775e-7（11.862s，非整条训练速度）。这说明旧≤128限制不适用于已经分离/物化梯度并隔离参数缓冲的新图；具体旧故障kernel尚未定位，未声称所有批次都长期稳定。GPU故障日志筛选15:18至查询时无新增hangcheck/gpu fault/offending task。

将当前计算块上界更新为已短测4096，推理块上界随设置增加至1024。1025环境/1024推理块（含1环境末块）对独立CPU公式PASS，最大误差2.861e-6；历史轨迹、参数更新/随机初始化/独立buffer均覆盖。B512真实仿真micro4096、inference1024、全JIT、主线程cpu7三轮测试进行中，代码快照与哈希已保存在运行目录。原PPO批次数和每轮更新仍是4和20。

2026-10-17 关键索引瓶颈：固定tinygrad0.14 mixin/op.py的Tensor高级索引以及gather均通过one-hot mask和源行归约实现；PPO每个microbatch的8个字段会扫描整份rollout，随总样本数与块大小乘积增长。旧索引B512 micro4096第3轮停滞，已停止；内核15:55后确认多次lockup，见onehot_b512_gpu_stall_kernel.log，不能把其约87% GPU busy当稳定性/有效算力通过。旧运行status手工标记failed_kernel_lockup，前两轮仅时间基线。

新增gpu_ops.take_rows，使用tinygrad UOp custom_kernel原生间接读取，保持GPU/JIT调度，输入rollout数据不需要梯度，不在CPU做gather。原型实机PASS：边界/重复索引、1D/2D/3D、float32位模式（含负零/NaN）、JIT新输入与历史输出独立；固定4096×26输出，源行1024到51200（50倍）耗时约10.26到11.64ms，不再全扫描。下一步优化线程组和完成完整PPO数值/实际训练回归。

2026-10-17 原生索引线程组优化：原型每线程组1lane效率低，改为128线程组与有界索引valid mask；中间STORE gate版本被tinygrad memory_coalescing拒绝，未执行GPU，改成index.valid后功能全PASS。固定4096×26输出，源1024/51200行实测约0.950/1.714ms，较原型10.26/11.64ms显著改善。native原B16×100 minibatch400/5epochs/20更新对照PASS4.775e-7；原FLARE逻辑minibatch51200、总204800样本、micro4096、1epoch/4更新对照PASS4.150e-6，含Adam状态/全局裁剪/自适应KL，GPU冷启动update12.824s。

同B512场景/种子/4k块、只改GPU索引的实测进行中：native前两轮33.444/21.990s（1530.9/2328.4 samples/s）；第一轮原始奖励与旧索引完全相同，第二轮仅约2e-9差异。旧索引前两轮124.337/108.801s且第3轮驱动故障，作为时间对照不作稳定性基线。将训练auto块改为min(逻辑minibatch,4096)，推理最大1024，原B16命令将自动整批400而非强制小块。默认并行数及JIT开关暂未切换，待最终性能/稳定性验证。

2026-10-17 Native实测：B512三轮全部PASS，后两轮2362.851 samples/s；B1024三轮PASS，后两轮约3370 samples/s；B2048（原版并行数）三轮PASS，后两轮约3773 samples/s，所有20更新/轮、100步/4逻辑批次与原版一致，内存余量充足。B2048 CPU-staging线程1对照也完成三轮，未见明确端到端提升，因此保留默认PyTorch线程设置；可选RL_TORCH_THREADS写入execution和manifest作移植调优，不改神经网络GPU训练。

根据实测将刚性默认并行设置为2048，解除早期B4临时限制；原柔性参考任务仍默认4（未做63自由度模型大规模测试），显式-B始终覆盖默认。scripts/run.sh默认启用梯度/Adam/rollout JIT与合并提交、主线程自动选择最高容量允许核；支持环境变量关闭作诊断，始终原生GPU行索引。README更新完整原配置对应关系、启动/续训/微块/执行开关/代码快照和移植检查。将用普通默认入口做10轮连续验证，之后补充最终结果与内核审计；这不是收敛或实飞验证。

2026-10-18 收尾（实验运行于10月17日）：默认2048入口10轮连续PASS，2048000样本/200次Adam更新，排除编译首轮3736.596 samples/s；峰值RSS1.014GiB、最低系统available4.516GiB、最高传感器74.3°C。B16实际续训10→11轮/200→220更新PASS，权重更新、模型/Adam有限且sigma为正，源checkpoint SHA保持不变。原PPO数值/随机数续训对照、404样本含64+37余块且8次实际全局裁剪、1025环境跨推理块/历史输出独立/bootstrap、日志统计回归均PASS。冻结GPU策略加载评估2个0.03秒测试回合PASS、权重不变，仅链路检查，非跟踪/收敛证据。README补充实测与限制，active_training更新默认配置；未启动长期训练。

内核审计完成：2026-10-17 16:00:00至2026-10-18 03:17:07，journalctl -k筛选hangcheck detected gpu lockup/gpu fault/offending task无匹配（exit1正常无结果），见native_gpu_kernel_audit.log。当前rl-*运行中服务0；不代表其他类型错误已全部排除。

2026-10-18 用户要求目录结构贴近原FLARE，并明确flare_q6a作为我们的工程入口。最终布局：flare_q6a/waypoint_passing_train.py、waypoint_passing_eval.py；内部envs/scenario1_rigid_env、genesis_bridge、surrogate_env；runners/ppo；utils配置/算子/硬件/日志；tests验证；config/envs与train原样复制固定上游YAML并从新位置加载。原同级Python入口保留转发兼容。根目录依赖子模块不改，assets/scripts/records保持原路径兼容。监控器使用新入口，源码快照递归包含新目录及兼容层。算法、参数、模型、记录目录与checkpoint schema保持不变。临时根目录代码布局已收回到flare_q6a，README更新目录树。

迁移检查发现直接执行入口时，sys.path包含flare_q6a，内部envs/utils常规包会遮蔽上游同名namespace包。已在两包__path__追加固定上游对应目录，保留原envs.scenario1_env和utils.mixer绝对导入；自身代码仍使用flare_q6a限定导入。首轮检查失败日志保留，修复后使用独立run_layout_train_check_fixed复验。

目录迁移期间旧check_env进程仍访问已移动临时根config，旧check_microbatch launcher仍引用临时tests路径，产生layout_surrogate_check/layout_ppo_check失败日志；不计作最终验证。最终限定包路径PPO对照layout_nested_ppo_check PASS，参数误差5.327e-7，404样本/64+37余块/8次实际裁剪与Adam更新。新旧静态入口、日志测试与AST/YAML一致性PASS。

最终flare_q6a目录GPU训练复验PASS：run_layout_train_check_fixed，4环境×4步×1轮，20次Adam更新，模型权重变化0.004976；状态completed，manifest源码快照包含入口/envs/runners/utils/config全部实际实现。小规模只验证迁移执行链路，不作性能或收敛结论。根目录 .gitignore 为用户原有修改，不纳入本次提交。

旧flare_q6a/evaluate.py入口加载迁移前第10轮checkpoint并在真实GPU完成2个0.03秒测试回合PASS，冻结权重不变，见layout_frozen_evaluation_check.json。仅检查兼容与执行，不作为跟踪效果。两份YAML保留原上游字节（含原行尾空格），git diff --check排除这两份导入文件后通过，其余代码检查无空白问题。

2026-10-18 用户要求根Git不跟踪原Flare工程：移除Flare gitlink和.gitmodules对应条目，新增/Flare/忽略规则，保留本机目录及原有.gitignore内容。原运行时仍依赖FLARE混控/柔性参考/资源，新增scripts/fetch_flare.sh按41f03b94固定提交获取并校验；setup自动调用，已有目录不覆盖。Genesis/rsl_rl_reference子模块不变，训练源码快照仍保存使用的上游文件。README说明外部依赖部署方式。

2026-10-18 用户要求用到及未用到的原FLARE内容均复制到q6a作参考：新增reference/Flare固定提交源码/配置/模型/网格/图片/文档快照及SOURCE.json逐文件SHA256，排除Git元数据/缓存/示例训练日志与checkpoint。实际运行独立复制envs/scenario1_env、utils/mixer、assets/suspended_system；刚性assets迁入q6a并更新视觉mesh相对路径。移除上游namespace扩展/sys.path/工作目录依赖，setup不再fetch原Flare，删除fetch脚本。训练快照递归包含全部本地源码与资源。刚性task_signature仅规范化迁移mesh路径，以兼容旧checkpoint；旧第10轮签名相同且所有模型mesh本地存在检查PASS。

独立性验证PASS：审计钩子禁止Python访问根Flare及q6a/reference，旧第10轮刚性策略GPU加载/2个0.03秒回合/权重不变PASS；柔性2环境GPU场景/reset/单步26维有限观测与奖励PASS。参考快照所有文件SHA256匹配来源清单，mixer逐字节一致，柔性环境除限定mixer导入外一致。保存check_standalone.py可复验。

最终监控入口真实GPU训练PASS：4环境×4步×1轮，20次Adam更新，权重变化0.004976，状态completed。manifest包含本地柔性环境/mixer/刚性URDF/网格/参考来源清单，没有根Flare文件依赖。原样复制的参考源码、mixer与模型网格保留原行尾空白，diff --check排除这些导入文件后通过；不为格式改变资源字节。

2026-10-18 用户要求先使用原柔性绳配置，控制变量比较高通GPU适配：训练/评估/监控入口默认任务改为s1_waypoint_passing，默认柔性并行4（原2048仅作为规模降低）；每环境100步、5epochs、4minibatches、800轮、原63DOF模型/观测/动作/奖励/航点/PID/mixer与策略采样噪声保持原样，不新增扰动或修正原控制器。刚性任务保留显式选项。文档区分柔性基线与历史刚性性能，旧刚性checkpoint不用于柔性续训。

用户补充柔性绳也加入q6a：确认实际运行envs/scenario1_env.py、assets/suspended_system/real_sus_exp.urdf及全部网格/材质已纳入q6a Git，非仅参考副本；增强对齐检查，柔性所有方法AST与原版一致、mixer/柔性URDF/环境与训练YAML逐字节一致。默认原配置训练使用这些本地资源。

默认无-e入口真实GPU柔性训练PASS：s1_waypoint_passing、4环境×100步、5epochs/4minibatches、1轮400样本/20次Adam更新，权重变化0.005031、checkpoint数值有限、状态completed；初次setup64.876s、训练30.043s（含首次GPU编译，不作稳态性能）。见flex_baseline_training_check.json/log。仅短流程验证，不是收敛或长期稳定结论；未启动长期训练。

2026-10-18 用户指出GPU并行已优化不应继续默认B4：B4仅是柔性模型临时未验证规模，并非当前GPU后端限制。此前2048实测属于刚性模型；现保持原柔性物理/控制/奖励/PPO，启动B128三轮原100步实测，按吞吐/内存/数值结果逐步扩大，最终按实测修订默认。

柔性扩容实测完成：B128/B512各3轮PASS、每轮100步与20次Adam更新，后两轮346.7808/1117.3610 samples/s，B512约3.22倍。两档checkpoint全部数值数组有限，模型sigma均正。柔性物理共享GPU内存显著高于刚性，见flexible_parallel_validation.json最低系统available；未尝试柔性2048，也不声明512为最大容量。根据实测将DEFAULT_FLEX_ENVS由4改为512，更新help、README、默认对齐断言与active_training；显式-B仍覆盖，原柔性控制/奖励/模型/PPO超参数未改。短程3轮不证明长期稳定/收敛，没有自动启动长期训练。
