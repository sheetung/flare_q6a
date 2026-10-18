# OpenCL驱动隔离对比

保持原柔性任务、B512、100步、5epochs/4minibatches、FP32、相同种子与执行开关；物理仍使用原Freedreno Vulkan ICD。先通过数值检查，再比较端到端训练，不把clpeak或单算子结果等同于RL吞吐。

## 准备专有运行库

```bash
bash scripts/prepare_qcom_opencl.sh
```

需要当前Ubuntu Qualcomm PPA中存在脚本锁定的四个包版本。只下载并用dpkg-deb解包到被Git忽略的records/run_opencl_driver_probe/，不运行apt install或包维护脚本，不替换系统库、默认ICD、内核和固件。packages.json记录包身份及SHA256；二进制包不提交到本仓库。

## 选择驱动

从工程根目录执行，选择仅对当前进程及其子进程有效：

```bash
bash scripts/opencl_driver.sh rusticl flare_q6a/utils/hardware.py
bash scripts/opencl_driver.sh qcom flare_q6a/utils/hardware.py

bash scripts/opencl_driver.sh qcom scripts/long_training.py \
  -B 512 --iterations 3 --output records/run_opencl_qcom_b512_compare
bash scripts/opencl_driver.sh rusticl scripts/long_training.py \
  -B 512 --iterations 3 --output records/run_opencl_rusticl_b512_compare
```

运行目录必须是新目录。默认scripts/run.sh仍使用原系统驱动；专有驱动通过独立OCL_ICD_VENDORS和LD_LIBRARY_PATH加载，显式指定系统OpenCL ICD loader。选择器不修改算法或精度。

## 解释结果

排除每次运行首次编译轮，按后两轮总样本/总耗时计算吞吐；分别报告采样与学习耗时。两种驱动的GPU随机数/初始化和浮点计算可能不同，因此不是逐动作、逐轨迹相同的配对实验，也不是收敛对比。固定种子和配置不能保证跨驱动完全相同的物理终止分布。

设备查询中专有驱动自报Adreno635，Mesa自报FD643；原样记录，两次运行使用同一块板，不据设备字符串判断更换了硬件。RSS不包含全部共享GPU内存，必须同时看系统available。DRM engine计数对两套驱动的覆盖可能不同，不把占用百分比当作等价性能指标。三轮仅为短程功能/吞吐验证，后续长时稳定、热稳态和多次交替运行仍需单独验证。

最终测量见records/opencl_driver_comparison.json；每次修改与验证归档到docs/CHANGELOG.md，并随代码提交本地Git。

## 本机结果（2026-10-18）

| 指标（排除首轮，后两轮） | RustiCL 25.2.8 | Qualcomm 0855.3 |
| --- | ---: | ---: |
| 吞吐（样本/秒） | 1028.9 | 1115.0 |
| 平均采样秒/轮 | 35.84 | 35.50 |
| 平均学习秒/轮 | 13.98 | 10.46 |
| 测试全程最低系统available GiB | 1.98 | 2.03 |
| 测试全程最高传感器°C | 72.0 | 81.7 |

本次配对专有驱动端到端吞吐高8.4%，PPO学习速度约1.34倍（学习耗时减少25.2%）。两组各3轮/60更新，源码SHA256、任务/模型签名、算法和执行配置一致，所有浮点checkpoint数组有限且float32、sigma为正。完整报告含逐项控制变量检查。

这是一次顺序短测，不是稳定收益证明。此前单独Rusticl B512测试后两轮为1117.4样本/秒，接近这次专有驱动1115.0，说明跨次运行波动不能忽略。温度起点/热历史未配对，最高传感器值也不同；未测功耗或热稳态。保持系统默认Rusticl，专有驱动作为可选对照；若要决定长期切换，需要多次交替跑与长时稳定性验证。

复现报告：

```bash
python3 scripts/compare_opencl_results.py \
  --qcom-run records/run_opencl_qcom_b512_compare \
  --rusticl-run records/run_opencl_rusticl_b512_compare \
  --output records/opencl_driver_comparison.json
```

数值检查还覆盖专有驱动51200样本、逻辑minibatch12800、4096 GPU块的一轮epoch/4次Adam更新，对照原rsl_rl最大参数误差3.450e-6，结果PASS，见records/qcom_full_b512_parity_check.log。
