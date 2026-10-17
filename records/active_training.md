# 当前后台长训练

- 服务：`rl-flare-long-20261017-02.service`
- 运行目录：`records/run_long_20261017_02`
- 开始：2026-10-17 04:14:32 UTC
- 训练源码 commit：`0927d959997498068604c1917db89a20a26f7b01`
- 原版 Genesis Vulkan + OpenCL PPO，8 环境 × 32 步 × 500 轮，5 epochs / 4 minibatches。
- 请求样本：128,000；每个 minibatch 64；每 5 秒监控；180 秒无进展看门狗。
- 训练 PID：14745，监督器 PID：14739（以 status.json 为准）。

2026-10-17 04:17:29 UTC 已完成 2 轮、40 次更新，检查点可读取，正在第 3 轮。最新采样 GPU 进程引擎占用约 33.4%，频率 812 MHz，进程 RSS 约 1.15 GiB。这不是完整长训练成功或收敛结论。

```bash
python3 scripts/training_status.py --output records/run_long_20261017_02 --watch
systemctl --user status rl-flare-long-20261017-02.service
```

日志：`training.log`；资源时序：`telemetry.jsonl`；当前资源：`latest_telemetry.json`；训练进度：`progress.json`；状态：`status.json`；恢复文件：`checkpoint.npz`。

NPU 未参与本次训练。GPU 利用率是本训练进程的 DRM 引擎时间差，不包括其他进程；GPU 频率单独记录。

前次 run_long_20261017_01 因 GPU lockup 已停止，没有检查点，日志保留；不能作为成功实验。
