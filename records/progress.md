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
