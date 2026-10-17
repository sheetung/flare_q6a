# 刚性杆负载悬停环境

独立实验名 `rigid_hover`；原 `s1_waypoint_passing` 不变。

目标：负载保持固定世界点，无人机位于负载正上方，杆竖直、负载速度接近零。默认负载目标 [0,0,0.52] m，杆长 0.48 m，对应无人机 [0,0,1] m。不是随无人机移动的相对目标，也不切换航点。

## 输入：19 维

世界坐标采用 ENU，机体采用 FLU；调用者提供同一世界坐标的状态，不在本环境实现 MAVROS 话题转换。

| 索引 | 输入 | 归一化 |
|---|---|---|
| 0–2 | 无人机位置减去负载目标+[0,0,杆长] | /1 m |
| 3–5 | 无人机世界速度 | /2 m/s |
| 6–9 | 无人机姿态，机体到世界的四元数 xyzw | 单位化、w非负 |
| 10–12 | 无人机机体系角速度 | /5 rad/s |
| 13–15 | 负载位置减去固定负载目标 | /1 m |
| 16–18 | 负载世界速度 | /2 m/s |

对应可读取的无人机局部位置/速度、IMU 姿态/角速度及动捕负载位置/速度。共享打包函数 `hover_observation.pack_hover` 可供之后部署使用；不额外喂入杆关节真值、负载姿态、上一动作。加速度计未包含在当前 19 维定义中。MAVROS 坐标参考：[官方 frames](https://github.com/mavlink/mavros/blob/ros2/docs/frames.md)、[IMU 插件](https://github.com/mavlink/mavros/blob/ros2/docs/plugins/std/imu.md)。具体消息有效性、时间同步、速度来源需在部署接入时处理。

## 动作与仿真

4 维 [-1,1]：总推力比例、机体系 roll/pitch/yaw 角速度指令。总推力比例=(a0+1)/2，最大推力为系统重量的 2.5 倍；角速度范围 ±[3,3,2] rad/s。仿真采用比例增益 0.1 的机体角速度内环与四桨混控，RPM 根据原模型 KF 换算。对称竖直悬停动作约 [-0.2,0,0,0]。这不是 PX4 控制器的精确复刻，实机部署需要推力及内环动态标定。

100 Hz，每个控制周期显式推进两次 5 ms 物理步，并每步重新施加保持不变的 RPM（此版本 Genesis 每个物理子步会清空外力），15 秒回合。初始位置±0.1 m、根部摆角±0.1 rad，速度清零。仅原生 Genesis Vulkan 物理与 Qualcomm OpenCL PPO，CPU 进行环境计算与数据搬运。模型结构见 ../assets/rigid_hover/README.md。

奖励为 dt 乘：2 exp(-(负载位置误差/0.25)²)、-0.05|负载速度|²、-0.5 sin²(摆角)、-0.1(1-Rzz)、-0.02|动作变化|²；失败另扣 1（不乘 dt）。触地、无人机离目标超过 2 m、倾角超过 75° 或摆角超过 80° 提前终止。超时与失败分开，保留 reset 前终止观测。

日志 `Hover criterion fraction` 表示每步同时满足负载误差<0.1 m、速度<0.2 m/s、摆角<10° 的比例，不能解释为完整回合成功率。另记录摆角、误差、杆长误差及每项奖励。

## 运行

```bash
cd /home/radxa/Documents/rl_radxa
# 默认自动创建带实验名/时间戳的目录，日志实时显示。
RL_PPO_BATCH_REALIZE=1 scripts/run.sh scripts/long_training.py \
  -e rigid_hover -B 8 --steps 32 --max_iterations 500
# 图形桌面加 -v 可请求仿真窗口；显示效果尚未实机验证。
# 测试刚性约束、推力平衡、超时及失败重置：
scripts/run.sh flare_q6a/check_rigid_hover.py --physics
# 加载检查点评估，将路径替换为本次生成目录：
scripts/run.sh flare_q6a/evaluate.py -e rigid_hover \
  --checkpoint records/run_rigid_hover_smoke_v2/checkpoint.npz --envs 2 --steps 100
```

先使用 8 环境作为新任务的起点；此前 16 环境性能结果属于旧柔性绳任务，不能直接当成新环境基准。请只启动一个 GPU 任务。旧 26 维航点策略与新 19 维悬停策略不兼容，需重新训练。检查点保存实验名称、观测定义与环境配置；续训环境 fresh reset。这里只完成环境与训练通路，尚未证明收敛或实机可用。

## 部分环境重置修复（reset_version: 2）

旧重置路径在仅部分环境失败时会留下旧负载位置、速度与根部广义坐标，导致重置后一步失败连锁；先前全批重置检查未覆盖此情况。已改为一次写入完整 qpos，然后对全部连续环境 ID 刷新运动学，绕过固定 Genesis 版本的部分索引刷新问题；未重置环境的物理状态保持不变。

同种子 16 环境×300 步随机动作，修复前后前104步完全相同，第105步首次正常越界；修复前1334次失败、1309次一步回合，修复后32次失败、0次一步回合、平均完成回合117.75步。见 ../records/rigid_hover_reset_comparison.json。随机动作试验不代表策略已经收敛。

旧 2cec0ab 长训练受到错误重置影响，保留记录，但应从头开始新训练；reset_version 写入配置，旧环境配置的 checkpoint 默认拒绝续训。训练命令不变。

```bash
scripts/run.sh flare_q6a/check_rigid_hover.py --physics --envs 16
scripts/run.sh flare_q6a/diagnose_rigid_hover.py --envs 16 --steps 300 --mode random \
  --output records/run_reset_diagnosis/retest.json
```

## 冻结策略均值评估

训练保持高斯动作探索。评估默认 `--mode mean`，只执行网络输出的均值，不更新网络；也可显式 `--mode sample` 作带探索的对照。请停止训练后再评估，避免两个 GPU 仿真任务争用。

```bash
scripts/run.sh flare_q6a/evaluate.py -e rigid_hover \
  --checkpoint records/<训练目录>/checkpoint.npz \
  --envs 16 --episodes 16 --mode mean --seed 123
```

默认自动创建 `records/run_eval_rigid_hover_<时间戳>/`。复制并冻结被测 checkpoint，记录 SHA256、轮数、随机种子、输入配置和策略模式；`episodes.jsonl` 保存逐回合时长、回报、误差、摆角、达标比例与终止原因，`evaluation.json` 保存汇总，`progress.json` 保存进度。评估结束验证权重未变化。结果的 status=PASS 表示评估程序检查通过，不表示策略已学会悬停。`--steps` 可限制物理控制步数；指定 episodes 时默认预算足够完成各环境配额，预算不足标为 INCOMPLETE，未结束轨迹明确记为 censored，不算成功。

`safe_full_duration_episodes` 只表示安全飞满15秒，不代表负载误差达标；同时检查 `mean_payload_error_m`、`mean_swing_deg` 和 `mean_hover_success`。任务均值按已完成回合的步数加权。`action_clip_fraction` 表示原始网络动作超出 [-1,1] 的分量比例。若比较 mean/sample，使用同一冻结 checkpoint 和相同 seed，但轨迹会因不同动作产生差异。

每次修改在 records/progress.md 记录原因、具体行为和验证结果，并提交本地 Git；运行时快照/逐步日志保留在忽略的 records/run_*/ 目录，关键测试结果另存为受 Git 管理的记录。
