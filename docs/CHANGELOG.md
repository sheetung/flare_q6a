# 修改与验证记录

每次修改在提交前追加本文件，与代码和验证报告一起提交。记录原因、涉及文件、验证结果及限制；提交身份以 `git log -- docs/CHANGELOG.md` 为准，不在同一提交内填写自身哈希。

## 2026-10-18：高通专有 OpenCL 与 RustiCL 对照（完成短程验证）

- 目的：固定原柔性 B512、100步、5epochs/4minibatches及FP32，比较两个OpenCL驱动；不调整物理、奖励、PPO或系统默认驱动。
- 方法：下载并解包 `qcom-adreno-cl1`、`qcom-adreno-common1`、`qcom-libdmabufheap`、`qcom-property-vault`，仅用进程环境选择独立ICD和库目录；不执行系统安装。
- 初步验证：专有驱动0855.3能创建设备及context，tinygrad识别Qualcomm vendor 0x5143；驱动自报设备名Adreno635，而Mesa为FD643，原样保存不据此推断硬件变化。
- GPU行索引精确位模式、边界/重复索引、JIT新输入和输出独立检查PASS。PPO余块/梯度裁剪/KL/Adam数值对照PASS（参数最大误差3.390e-7）；实际B512逻辑minibatch12800/4096块的大批次对照也PASS（最大参数误差3.450e-6、4次Adam更新）。两驱动固定柔性B512各3轮/60更新通过，checkpoint全部浮点数组有限且FP32、sigma正。
- 记录：`records/qcom_opencl_probe.log`、`qcom_tinygrad_device_probe.log`、`qcom_gpu_ops_check.log`。
- 文件：新增scripts/prepare_qcom_opencl.sh、opencl_driver.sh、compare_opencl_results.py及docs/OPENCL_COMPARISON.md；long_training.py补充驱动选择字段和选择器源码快照；README增加记录入口。
- 结果：后两轮Rusticl 1028.9、QCOM 1115.0样本/秒，本次+8.4%；学习13.98→10.46秒/轮。控制变量及源码哈希一致，报告records/opencl_driver_comparison.json。
- 限制：只做一次顺序短测；此前Rusticl独立测试1117.4样本/秒，存在跨次波动。温度/热历史未控制，不作为稳定收益、收敛或长期稳定结论。系统默认Rusticl不变，专有驱动保留可选。
- 归档：clinfo日志仅清理行尾空格，不改设备信息或数值；代码语法和Git空白检查通过。
- 环境：系统未安装qcom包，/etc/OpenCL/vendors仍只保留原rusticl.icd；运行库二进制不提交，包版本/SHA256记录在records/qcom_opencl_runtime_packages.json。

## 2026-10-18：最近已完成提交补录

| 提交 | 修改 | 验证 |
| --- | --- | --- |
| `1315cce` | 柔性默认并行由临时4改为实测512 | B128/B512各3轮、60次Adam更新通过；后两轮346.8/1117.4样本每秒；数值有限 |
| `3163bbf` | 默认恢复原FLARE柔性任务，保留原混控/模型/奖励/PPO | B4×100步一轮/20次Adam更新通过；原文件/方法对齐 |
| `6b53866` | 运行模块和模型资源复制进q6a，原工程源码资源保存为参考快照 | 禁止访问外部Flare/reference的刚性旧策略评估与柔性步进通过；训练/资源快照通过 |
| `ebc5535` | 根Flare取消Git跟踪并忽略 | 本机文件保留、固定版本及对齐检查通过；外部依赖方案随后由6b53866取代 |
| `4da043c` | q6a内部按FLARE整理envs/config/runners/utils/tests | GPU训练、数值对照、旧检查点评估通过；旧入口保留兼容 |
| `0827d47` | 归档原生GPU适配默认值及稳定性报告 | 刚性B2048连续10轮、2048000样本/200更新通过；B16续训通过 |

更早的修改和实验细节保留在 `records/progress.md` 与 Git 历史中。短程验证不等于长期稳定、收敛或实飞证明。

## 2026-10-18：用户B1024柔性训练低GPU计数排查

- 只读检查运行run_s1_waypoint_passing_20261018_043415_108058，未停止或修改训练。
- 确认OpenCL GPU/全部JIT/native gather启用，GPU812MHz；日志GPU数值是最近约5秒进程DRM engine窗口，非整轮平均或ALU占用。
- 用户第2轮：采样46.19s、学习20.73s，1531.3样本/秒；B1024吞吐仍高于此前B512短测，不能说低计数代表优化无效。
- 观测系统available约0.5–0.6GiB、swap约5.1GiB；vmstat短窗口出现3580/1396KiB每秒换出，说明当前确有内存压力，不能仅用进程RSS评估GPU共享内存。
- 记录records/b1024_live_memory_diagnosis.json。建议长期实验优先保留B512内存余量；不宣称它比B1024更快，也不把这次实时观察当受控性能对比。未修改算法、默认值或当前任务。

## 2026-11-01：整理测试并准备推送远程

- 按用户要求整理工程：移除一次性gpu_probe探针、重复的flare_q6a/check_*.py转发入口，以及仅对手写公式作重复验证的check_ppo.py；核心PPO原版数值与续训、微批、GPU索引、物理对齐、日志和部署检查保留在flare_q6a/tests/。
- 测试命令统一使用flare_q6a/tests/路径；训练和评估旧入口继续兼容。保留records中已跟踪的实验、历史诊断及用户本机忽略的训练目录。
- README更新为用户指定的GitHub仓库地址和测试目录；原环境、模型、PPO参数、默认训练命令及子模块固定版本不变。
- 验证：环境对齐、日志/评估统计、入口help、语法和Git差异检查通过；实机Adreno GPU行索引、原版PPO与续训对照、微批余块及8次实际梯度裁剪均通过，见records/cleanup_validation.json。

## 2026-11-01：配置 SSH 远程并推送

- 按用户提供的 SSH 地址配置 origin，使用现有 SSH 密钥完成 GitHub 身份验证，main 已设置跟踪 origin/main。
- 当前网络直连 GitHub 22 端口超时；此工作区的本地 Git 配置使用现有代理连接 GitHub SSH 443，主机公钥来自 GitHub 官方元数据并严格校验。代理设置仅保存在本机 .git 中，不随仓库发布。
- 验证：首次推送后，通过 SSH 读取远端 main，确认与本地提交一致；更新 README 使用 SSH 克隆地址。

## 2026-11-01：更新仓库地址与策略迁移流程

- 将本地 origin 更新为 `git@github.com:sheetung/flare_q6a.git`，SSH 读取远程 main 成功；精简 README，主流程改为 NVIDIA 训练、板端迁移和 GPU 仿真评估，板端训练保留作调试。
- 增加原版 `.pt` 与可信 `config.yaml` / `cfgs.pkl` 配置导入、OpenCL 检查点及确定性 Actor ONNX 导出工具；不包含优化器，不能用于续训；新增仿真观测采样与 QAIRT 转换/量化调用脚本。
- 原版第 800 轮公开模型在 Adreno 643 上转换核对，Actor 最大绝对误差 9.54e-7、Critic 4.77e-7；ONNX 与 PyTorch 最大绝对误差 1.49e-7。原版模型下载的 SHA256 核对成功；拒绝覆盖已有输出的检查通过。
- GPU 仿真种子 123、4 环境 4 回合：3 失败、1 回合 15.01 秒超时，共 63 航点；仅证明执行完成，不代表稳定。日志记录在 records/pretrained_flare_evaluation.log。
- QAIRT 系统软件包隔离检查发现转换器缺少 `qti.aisw` 模块；完整 SDK 转换、量化、NPU 运行及 PX4 控制尚未验证。流程、接口和限制见 docs/MODEL_MIGRATION.md。
- 新入口迁移和原始仿真观测采集实机通过（2 环境、3 步，6 条观测仅用于流程测试）；QAIRT 参数串联使用模拟命令验证，不代表真实量化通过。原始观测列表跨主机复制后需重新生成绝对路径。
