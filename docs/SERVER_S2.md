# 服务器 S2 刚性场景

服务器 `192.168.31.10` 的 `/home/sheetung/flare_ws` 在本次检查时为提交 `d99dabb234da32ec255c57e4d9ca3d5dca1bdc33`，实际显卡为 RTX 5060 Ti 16GB。选择 `s2_waypoint_passing_rigid`：2 个绳根转动关节、26 维观测、4 维动作，Actor/Critic 均为 `[256,256,128]`、ELU。各场景由各自模型及配置文件定义，不要将此前的本机 `s1_waypoint_passing_rigid` 视为此模型。

- 环境：`flare_q6a/envs/scenario2_server.py`
- 模型：`flare_q6a/assets/server_s2/real_sus_exp_rigid.urdf`
- 环境配置：`flare_q6a/config/envs/server_s2_rigid.yaml`
- PPO 配置：`flare_q6a/config/train/server_s2_rigid.yaml`
- 混控：`flare_q6a/utils/server_s2_mixer.py`

本机使用 GsTaichi Vulkan 物理与 Adreno OpenCL 网络，部分观测、混控和数据传递由 CPU 完成。为处理本机 Genesis 部分重置后链接变换残留，S2 本机环境采用完整 qpos 重置并刷新所有环境的运动学；服务器源码对应的重置方式不同，比较实验应考虑这一差异。本机物理引擎提示部分关节摩擦参数不受支持，也需要纳入跨设备差异。

## NVIDIA 训练与连接

```bash
cd /home/sheetung/flare_ws/Flare
../.venv/bin/python -u train.py -e s2_waypoint_passing_rigid -B 2048 --max_iterations 800 --device cuda:0
```

本次正式训练运行在服务器 `tmux` 会话 `q6a_rigid_800`。本机的 `scripts/watch_server_s2.sh` 会在连接中断后重试并显示当前训练轮次；该监控脚本不会自动重启训练。`scripts/finish_server_s2.sh` 针对本次指定的训练目录等待完成并自动重连，核对远端和本机文件 SHA256 后导入和评估；本次在本机 tmux 的 `q6a_s2_auto_finish` 会话中运行。SSH 专用私钥和首次获取的主机密钥位于本机被 Git 忽略的 `records/run_ssh_5070ti/`；不要公开私钥。首次连接使用获取到的主机密钥固定连接，尚未通过独立渠道核验其指纹。

两机时区均为 Asia/Shanghai；已按本机校准服务器时钟并启用 NTP。检查时两机系统均显示 NTP 尚未同步，不能保证长期不漂移。校时前的服务器正式训练已按要求删除，并在校时后重新开始；本机留存的旧实验副本不属于新训练。

## 本机训练与 NVIDIA 策略评估

```bash
scripts/run.sh scripts/long_training.py -e s2_waypoint_passing_rigid -B 4

# 将同一服务器训练目录中的 model_800.pt 和 cfgs.pkl 复制到本机后：
scripts/run.sh scripts/import_server_s2.py --checkpoint /path/to/model_800.pt \
  --config /path/to/cfgs.pkl --output records/run_s2_transfer/policy.npz
scripts/run.sh scripts/eval_server_s2.py --checkpoint records/run_s2_transfer/policy.npz \
  --output records/run_s2_evaluation --envs 4 --episodes 4 --steps 1501
```

仅加载可信来源的 `.pt` 和 `cfgs.pkl`，后者为可执行代码的 pickle。导入器核对配置、网络结构和 Actor 数值，并记录任务签名；生成的文件只包含 Actor，用于推理而不能续训。评估器也支持同一 S2 本机训练生成的 `checkpoint.npz`，并检查任务签名。模型、配置、环境源码或网格变化后需要重新导入，不能直接混用不同场景的 checkpoint。

评估统计完整回合和被截断回合；执行完成不代表策略稳定或能够实飞。工程尚未接入 PX4 控制链。
