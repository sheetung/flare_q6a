# NVIDIA 训练模型迁移至 Q6A

## 瑞莎官方资料

- [Dragon Q6A 官方教程](https://docs.radxa.com/dragon/q6a)和 [OpenCL 使用指南](https://docs.radxa.com/dragon/q6a/system-config/opencl)。
- [启用 NPU](https://docs.radxa.com/en/dragon/q6a/app-dev/npu-dev/fastrpc-setup)、[NPU 快速验证](https://docs.radxa.com/en/dragon/q6a/app-dev/npu-dev/quick-example)。
- [QAIRT 安装](https://docs.radxa.com/en/dragon/q6a/app-dev/npu-dev/qairt-install)、[模型转换及 NPU 推理示例](https://docs.radxa.com/en/dragon/q6a/app-dev/npu-dev/qairt-usage)。安装文档也提供 QCS6490 ARM64 Docker 镜像，是否适用于本策略需实测。

官方示例的 QCS6490 HTP 参数为 `dsp_arch=v68`、`soc_id=35`，量化 DLC 后还需生成 Context-Binary，再在设备端运行。具体 SDK 和驱动版本以官方教程及实际系统为准。

## 环境准备

NVIDIA 训练端使用 [FLARE 固定版本](https://github.com/BEI11HAI/Flare/tree/41f03b94f2f724205e5e24617010467aae2a3f98)及其指定 Genesis / rsl_rl 版本。原训练脚本将配置保存为 `cfgs.pkl`，与检查点一起复制。仅加载可信文件：Python pickle 可执行任意代码；`model_*.pt` 也应来自可信来源。

Q6A 已验证 Ubuntu 24.04 ARM64、Python 3.12、Adreno 643。安装 GPU 系统依赖后执行 `scripts/setup.sh`；其他板卡须验证厂商驱动。

```bash
sudo apt-get update
sudo apt-get install -y python3.12-venv python3.12-dev build-essential git \
  mesa-opencl-icd clinfo vulkan-tools libgl1 libegl1 libglfw3
RUSTICL_ENABLE=freedreno clinfo -l
vulkaninfo --summary
```

OpenCL 和 Vulkan 必须显示真实 GPU，不接受 llvmpipe 软件设备。参考 [Radxa OpenCL](https://docs.radxa.com/dragon/q6a/system-config/opencl)。

## 官方示例模型

原项目公开的 `model_800.pt` 与 `config.yaml` 可以用于验证迁移：

```bash
mkdir -p records/run_official_flare_example
base=https://raw.githubusercontent.com/BEI11HAI/Flare/41f03b94f2f724205e5e24617010467aae2a3f98/logs/s1_waypoint_passing
curl -fL "$base/model_800.pt" -o records/run_official_flare_example/model_800.pt
curl -fL "$base/config.yaml" -o records/run_official_flare_example/config.yaml
sha256sum records/run_official_flare_example/{model_800.pt,config.yaml}
scripts/migrate_flare_policy.sh records/run_official_flare_example/model_800.pt records/run_example_policy
```

`model_800.pt` SHA256：`b0d09610c5f0e3c8a4b9224ed469ad69b1014759fb6bfe415392487888110e8a`；`config.yaml` SHA256：`5326a9546ced8ad222f1b312b9bc398ab6c46cab5ed0d0623cd1139fe6cc5455`。下载并人工核对哈希后再运行迁移。两种配置格式均可自动识别，也可通过第三个参数显式指定。

迁移工具核对环境、观测、奖励和目标配置以及网络结构；仅接受匹配当前柔性任务的策略。输出用于推理，不含优化器，不能用于续训。转换时以原 PyTorch 和板端 OpenCL 分别计算 Actor、Critic 并比较；ONNX 与 PyTorch 再比较。若流程失败，删除自行确认的未完成输出目录后重试。

## 模型接口与验证范围

输入 `observation` 为 FP32 `[1,26]`：当前和下一航点误差、机体速度、姿态矩阵、前一动作、绳角及角速度，缩放和顺序见 `flare_q6a/envs/scenario1_env.py`。输出 `action_mean` 为 FP32 `[1,4]`，是油门和三轴角速度目标；ONNX 不含动作裁剪，调用方须按原环境裁剪至 `[-1,1]` 并执行后续映射、PID 和混控。本工程目前没有 PX4 接口，也没有完成机体系、坐标系、控制频率和飞行安全验证。

本机原示例模型转换：Actor 与 PyTorch 最大绝对误差约 `9.54e-7`，Critic 约 `4.77e-7`，ONNX Actor 与 PyTorch 约 `1.49e-7`。种子 123、4 环境、均值策略评估中，4 回合 3 次失败，1 次达到 15.01 秒上限；共触达 63 个航点。这是一次小样本运行，不能证明稳定或实飞可用，评估 `PASS` 仅表示执行完成。记录见 `records/pretrained_flare_evaluation.log`。

## QAIRT 量化与 NPU

参考 [Radxa QAIRT 安装](https://docs.radxa.com/en/dragon/q6a/app-dev/npu-dev/qairt-install)和[转换与 NPU 示例](https://docs.radxa.com/en/dragon/q6a/app-dev/npu-dev/qairt-usage)。`collect_policy_calibration.py` 保存模拟环境中的真实策略输入，每条为 26 个 FP32；应使用足够多、覆盖部署状态的数据，单次仿真采样不能证明代表性。`quantize_policy_qairt.sh` 调用完整 SDK 的 `qairt-converter` 和 `qairt-quantizer`，生成 DLC。

Radxa 的示例将转换和量化放在 x86 Linux 主机上。将 `actor.onnx`、采集目录和 `scripts/quantize_policy_qairt.sh` 复制到该主机，并在复制后的校准目录中重建绝对路径列表，再运行 README 的量化命令。例如：

```bash
python3 - /path/to/calibration <<'PYCODE'
from pathlib import Path
import sys
folder = Path(sys.argv[1]).resolve()
files = sorted(folder.glob('*.raw'))
if not files:
    raise SystemExit('No raw calibration inputs')
(folder / 'input_list.txt').write_text(''.join(f'{p}\n' for p in files))
PYCODE
```

当前系统找到运行库和工具包，但隔离检查的工具缺少 `qti.aisw` 转换模块；完整 SDK、脚本与目标板兼容性、DLC 量化质量和实际 NPU 推理尚未验证。生成 DLC 也不等于完成飞控部署；需按 SDK 和板卡要求生成或加载 HTP 上下文，在相同观测集对比 FP32 与量化输出，并进行仿真闭环验证。
