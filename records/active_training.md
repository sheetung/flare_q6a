# 当前任务：原FLARE刚性吊挂航点跟踪

实验 s1_waypoint_passing_rigid，默认2048环境×100步，原26维观测、六项奖励、目标、PID与PPO。独立模型 assets/rigid_suspended_system/rigid_suspended_system.urdf；柔性参考默认4环境。

默认入口10轮连续验证PASS：2,048,000样本、200次Adam更新，稳定阶段3736.6样本/秒。有效检查点 records/run_native_default_stability/checkpoint.npz（第10轮）。B16追加一轮续训也PASS；各次运行保存独立源码快照。

本轮没有启动长期训练。启动命令见根README；默认800轮，每100轮及正常结束保存checkpoint。10轮仅为有限稳定性验证，不是收敛或实飞验证；冻结加载测试仅0.03秒回合。
