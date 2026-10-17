# 当前任务：原FLARE刚性吊挂航点跟踪

默认实验 s1_waypoint_passing_rigid，4环境×100步，原26维观测/奖励/目标/PID/PPO。独立模型 assets/rigid_suspended_system/rigid_suspended_system.urdf。自定义悬停配置与代码已删除，历史运行记录保留。

静态及Vulkan物理对齐检查已通过，记录见 rigid_waypoint_alignment_checks.log。下一步运行4×100两轮训练验证；没有自动启动长期训练。默认每100轮及正常结束保存checkpoint，旧悬停checkpoint不能续训此任务。
