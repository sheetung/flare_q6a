# 当前任务：原FLARE柔性绳航点跟踪基线

默认s1_waypoint_passing，4环境×100步、5epochs、4minibatches、800轮。原柔性模型63内部DOF、混控/PID、26维观测、4维动作、奖励与目标均沿用固定上游。仅后端改为Vulkan物理/OpenCL PPO，GPU块/JIT按适配配置执行。

刚性任务保留为显式-e s1_waypoint_passing_rigid，历史2048并行性能只属于刚性任务。原刚性checkpoint不能恢复到柔性任务。没有自动开启长期训练。

默认启动：scripts/run.sh scripts/long_training.py
