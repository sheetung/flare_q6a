# FLARE on Qualcomm

在 NVIDIA GPU 上训练 FLARE 无人机吊挂策略，再迁移到 Radxa Dragon Q6A 等高通平台验证和部署。当前主要支持原版柔性绳 `s1_waypoint_passing` 航点跟踪；刚性绳任务也保留在 `flare_q6a/`。

已验证 Q6A 的 Adreno GPU 策略推理和 Vulkan 仿真。NPU 的量化脚本已准备，完整 QAIRT 转换、量化和 NPU 推理尚待验证。工程尚未接入 PX4 实飞控制。

## 1. NVIDIA 上训练

按照 [原 FLARE 安装说明](https://github.com/BEI11HAI/Flare/tree/41f03b94f2f724205e5e24617010467aae2a3f98#installation) 配置 NVIDIA 训练环境，然后运行：

```bash
python waypoint_passing_train.py -e s1_waypoint_passing -B 2048 --max_iterations 800
```

训练完成后，将同一日志目录中的 `model_800.pt` 和 `cfgs.pkl` 一起复制到 Q6A。也可以使用原项目提供的示例模型；来源和校验方法见[迁移文档](docs/MODEL_MIGRATION.md)。只使用可信来源的模型和配置文件。

## 2. Q6A 上迁移并评估

```bash
git clone --recurse-submodules git@github.com:sheetung/flare_q6a.git
cd flare_q6a
# 根据板卡安装 OpenCL/Vulkan 驱动及 Python 3.12 等依赖，详见迁移文档
scripts/setup.sh
scripts/run.sh -m pip install --no-cache-dir --no-deps --target records/run_onnx_export_tools onnx==1.16.2

# 替换成自己的模型路径；输出目录必须是新目录
scripts/migrate_flare_policy.sh /path/to/model_800.pt records/run_my_policy
scripts/run.sh flare_q6a/waypoint_passing_eval.py \
  --checkpoint records/run_my_policy/policy.npz \
  -e s1_waypoint_passing --envs 4 --episodes 4 --mode mean
```

迁移生成 GPU 用的 `policy.npz` 和仅含确定性 Actor 的 `actor.onnx`。评估结果自动保存在 `records/`。板端训练可用于调试，但由于资源限制不推荐作为主要训练方式。

## 3. 准备 NPU 量化

先在 Q6A 采集仿真观测。按 [Radxa 教程](https://docs.radxa.com/en/dragon/q6a/app-dev/npu-dev/qairt-usage)，转换与量化可在配置完整 QAIRT SDK 的 x86 Linux 主机进行：

```bash
scripts/run.sh scripts/collect_policy_calibration.py \
  --checkpoint records/run_my_policy/policy.npz \
  --output-dir records/run_my_calibration --envs 4 --steps 500

# 将 actor.onnx、校准目录和量化脚本复制到 QAIRT 主机；重新生成绝对路径列表后执行
scripts/quantize_policy_qairt.sh records/run_my_policy/actor.onnx \
  records/run_my_calibration/input_list.txt records/run_my_quantized
```

量化后还需要验证误差、NPU 运行和闭环控制；采样数据应覆盖目标运行场景。设备适配、模型接口、示例模型和当前验证结果见[模型迁移文档](docs/MODEL_MIGRATION.md)。每次工程修改记录在[更新日志](docs/CHANGELOG.md)。
