# FLARE on Qualcomm

在 NVIDIA GPU 上训练 FLARE 无人机吊挂策略，再迁移到 Radxa Dragon Q6A 等高通平台验证和部署。场景使用 `s*` 命名，并分别保存环境、模型和训练配置。

板端使用 Adreno GPU 进行策略推理，使用 Vulkan 进行仿真；已完成模型转换和仿真运行验证。工程尚未接入 PX4 实飞控制。

当前场景：

| 场景 | 模型与配置 |
| --- | --- |
| `s1_waypoint_passing` | 原 FLARE 柔性绳 |
| `s2_waypoint_passing_rigid` | 服务器对应的双关节刚性绳航点任务 |

其他服务器场景需逐项迁移和验证；旧的 `s1_waypoint_passing_rigid` 名称保留兼容，它与 `s2` 不是同一个模型。

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

# 替换成自己的模型路径；输出目录必须是新目录
scripts/run.sh scripts/import_flare_policy.py \
  --checkpoint /path/to/model_800.pt --output records/run_my_policy/policy.npz
scripts/run.sh flare_q6a/waypoint_passing_eval.py \
  --checkpoint records/run_my_policy/policy.npz \
  -e s1_waypoint_passing --envs 4 --episodes 4 --mode mean
```

迁移生成 GPU 用的 `policy.npz`，无需量化。评估结果自动保存在 `records/`。板端训练可用于调试，但由于资源限制不推荐作为主要训练方式。

本机按相同场景名启动 S2 训练（默认 4 个并行环境）：

```bash
scripts/run.sh scripts/long_training.py -e s2_waypoint_passing_rigid
```

服务器 S2 模型的导入与评估命令见 [S2 迁移记录](docs/SERVER_S2.md)。板卡配置参考 [Radxa Q6A 官方教程](https://docs.radxa.com/dragon/q6a)。模型接口、示例模型和验证结果见[模型迁移文档](docs/MODEL_MIGRATION.md)；每次修改记录在[更新日志](docs/CHANGELOG.md)。
