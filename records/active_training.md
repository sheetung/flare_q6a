# 当前训练状态（2026-10-17 GPU 测试完成）

本轮 GPU 测试全部结束，未启动新的长训练。16×32 合并提交版本三轮及 GPU 恢复后两轮均完成，32×32/minibatch 256 因内核 GPU lockup 停止。结果见 gpu_benchmark_results.json。旧用户训练及其检查点保留；下方为历史快照。

# 历史后台长训练 03（已停止）

- 服务：`rl-flare-long-20261017-03.service`
- 运行目录：`records/run_long_20261017_03`
- 启动 UTC：2026-10-17T04:53:54.642865+00:00
- 实验源码 commit：`276497b45f692a46e650085286a23645daf79edd`
- 训练 PID：16625；监督器 PID：16622（以 status.json 为准）。
- 原版 Genesis Vulkan + OpenCL PPO，8 环境 × 32 步，5 epochs / 4 minibatches。
- 从旧 run_long_20261017_02 第 49 轮 checkpoint 恢复，追加 451 轮至总第 500 轮。
- 每 5 秒监控，180 秒无训练进展看门狗；NPU 未参与。

静态快照：已记录第 53 轮，完整回合窗口 11 个；平均回报 -0.3062300020127292，平均回合长度 64.0 步。最新进度以 progress.json 为准，此文不实时刷新。

```bash
python3 scripts/training_status.py --output records/run_long_20261017_03 --watch
tail -f records/run_long_20261017_03/training.log
systemctl --user status rl-flare-long-20261017-03.service
```

日志：training.log；结构化指标：metrics.jsonl / metrics.json；TensorBoard：tensorboard/；资源时序：telemetry.jsonl；当前资源：latest_telemetry.json；恢复文件：checkpoint.npz。

日志包含完整回合统计、六项奖励分解、航点到达数、位置误差、摆角、终止原因、FPS、KL、动作噪声和 ETA。GPU 利用率是本进程 DRM 引擎时间差，不含其他进程。

旧 run_long_20261017_02 已在第 49 轮检查点保存后停止，旧日志保留。完整物理状态未恢复，环境 fresh reset；旧日志没有完整回合明细，无法补造，新的回合窗口从恢复后开始积累。

更早的 run_long_20261017_01 因 GPU lockup 停止，没有检查点，不是成功实验。
