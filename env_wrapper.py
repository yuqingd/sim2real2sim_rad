from dm_control import suite
from metaworld.envs.mujoco import env_dict as ed

from mw_wrapper import MetaWorldEnv


def make(domain_name, task_name, seed, from_pixels, height, width, cameras=range(1),
         visualize_reward=False, frame_skip=None):
    if domain_name in suite._DOMAINS:
        import dmc2gym
        env = dmc2gym.make(
            domain_name=domain_name,
            task_name=task_name,
            seed=seed,
            visualize_reward=visualize_reward,
            from_pixels=from_pixels,
            height=height,
            width=width,
            frame_skip=frame_skip
        )
        return env
    elif domain_name in ed.ALL_V1_ENVIRONMENTS.keys():
        env_class = ed.ALL_V1_ENVIRONMENTS[domain_name]
    elif domain_name in ed.ALL_V2_ENVIRONMENTS.keys():
        env_class = ed.ALL_V2_ENVIRONMENTS[domain_name]
    else:
        raise KeyError("Domain name not found.")

    if task_name is not None:
        env = env_class(task_type=task_name)
    else:
        env = env_class()

    env = MetaWorldEnv(env, from_pixels=from_pixels, cameras=cameras, height=height, width=width)
    env.seed(seed)
    return env


def generate_shell_commands(domain_name, task_name, cameras, observation_type, encoder_type, work_dir,
                            pre_transform_image_size, agent, data_augs=None,
                            seed=-1, critic_lr=1e-3, actor_lr=1e-3,
                            eval_freq=10000, batch_size=128, num_train_steps=1000000, cuda=None,
                            save_tb=True, save_video=True, save_model=False):
    if cuda is not None:
        command = 'CUDA_VISIBLE_DEVICES=' + cuda + ' python train.py'
    else:
        command = 'python train.py'
    command += ' --domain_name ' + domain_name
    if task_name is not None:
        command += ' --task_name ' + task_name
    command += ' --cameras ' + cameras
    command += ' --observation_type ' + observation_type
    command += ' --encoder_type ' + encoder_type
    command += ' --work_dir ' + work_dir
    command += ' --pre_transform_image_size ' + pre_transform_image_size
    command += ' --image_size 84'
    command += ' --agent ' + agent
    if data_augs is not None:
        command += ' --data_augs' + data_augs
    command += ' --seed ' + str(seed)
    command += ' --critic_lr ' + str(critic_lr)
    command += ' --actor_lr ' + str(actor_lr)
    command += ' --eval_freq ' + str(eval_freq)
    command += ' --batch_size ' + str(batch_size)
    command += ' --num_train_steps ' + str(num_train_steps)
    if save_tb:
        command += ' --save_tb'
    if save_video:
        command += ' --save_video'
    if save_model:
        command += ' --save_model'
