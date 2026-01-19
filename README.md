loco-mujoco-download-real命令下载的数据集存放在: /home/robot/RL/loco-mujoco-0.4.1/loco_mujoco/datasets/humanoids/real/

当前locomujoco版本：0.4.1

当前项目为修改CMU项目得到，去除数据合成，训练单一速度骨骼模型

目前在使用CPU训练？修改模型权重保存逻辑（保存最优模型及定期保存模型）

1.25 m/s 单速训练 使用locomujoco提供的专家数据 -> 更换专家数据(KIT数据集 + 重定向)

模仿学习算法：VAIL+TRPO -> 修改AMP + PPO

任务：locomujoco自带walk任务 -> 自定义steering任务

TODO:

先测试一下代码还能不能跑，别删文件删错了。
参考新版locomujoco源码 将项目重构，对齐新版locomujoco接口
新版locomujoco提供AMP及PPO实现 参考AmpSteering项目实现steering任务（考虑怎么办，最好不要修改源代码)
新版locomujoco提供AMASS重定向 替换用于训练的专家数据
