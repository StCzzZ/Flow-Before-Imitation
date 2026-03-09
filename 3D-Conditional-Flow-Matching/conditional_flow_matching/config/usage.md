# Change To A New Object:
1. chage the zarr path in the task configuration
2. change the task name(s) inside the task configuration
3. change the object name inside the isaaclab env configuration, like "shadow_hand_env_cfg.py"
4. train


# About tasks

pyramid, ball, double ball: fix wrist

other objs(cube, vase, apple), not fixed

pyramid需要固定手腕的原因是,如果不固定的话它会转到一个位置然后把手腕翻下去来匹配pose, 这样点云就啥也看不见了