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
