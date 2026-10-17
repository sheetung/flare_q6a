# 当前任务：原FLARE刚性吊挂航点跟踪

实验 s1_waypoint_passing_rigid，默认4环境×100步，原26维观测/六项奖励/目标/PID/PPO。独立模型 assets/rigid_suspended_system/rigid_suspended_system.urdf。自定义悬停代码与配置已删除，历史记录保留。

静态和实际Vulkan对齐检查PASS；4×100两轮训练PASS（800样本、40次更新），最终检查点 records/run_rigid_waypoint_train_check/checkpoint.npz；检查点加载及强制超时评估PASS，非收敛结果。记录见 rigid_waypoint_alignment_checks.log、rigid_waypoint_training_check.json、rigid_waypoint_evaluation_check.json。

没有自动开启长期训练。启动命令见根README；默认800轮，每100轮及正常结束保存checkpoint。旧悬停checkpoint不兼容。
