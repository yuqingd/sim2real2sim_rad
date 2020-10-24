import argparse


def generate_shell_commands(domain_name, task_name, run_type, save=False, action_repeat=None, alpha=.1,
                            seed=1, start_outer_loop=5000,
                            eval_freq=2000, scale_large_and_small=True, delay_steps=2, time_limit=60,
                            mean_scale=None, prop_range_scale=True, prop_train_range_scale=True, sim_param_layers=3,
                            sim_param_units=512, separate_trunks=True, train_range_scale=1, range_scale=1,
                            dr_option=None, update_sim_param_from='buffer', num_eval_episodes=2,
                            num_sim_param_updates=3, continue_train=False):
    command = 'CUDA_VISIBLE_DEVICES=X python train.py --gpudevice X --id Y'
    command += ' --domain_name ' + domain_name
    command += ' --task_name ' + task_name
    command += ' --seed ' + str(seed)
    command += ' --save_video'
    command += ' --save_tb'
    command += ' --use_img'
    command += ' --start_outer_loop ' + str(start_outer_loop)
    command += ' --num_eval_episodes ' + str(num_eval_episodes)
    command += ' --eval_freq ' + str(eval_freq)
    if continue_train:
        command += ' --continue_train'

    if save:
        command += ' --save_model'
        command += ' --save_buffer'

    # Defaults specific to the type of run (OL3, oracle, baseline)
    if run_type == 'oracle':
        command += ' --outer_loop_version 0'
    elif run_type == 'baseline':
        mean_scale = 2
        command += ' --outer_loop_version 0'
    elif run_type == 'BCS':
        command += ' --outer_loop_version 0'
        mean_scale = 1
    elif run_type == 'OL3':
        mean_scale = 2
        if continue_train:
            range_scale = .1
            train_range_scale = .1
        command += ' --outer_loop_version 3'
        command += ' --prop_alpha'
        command += ' --update_sim_param_from ' + update_sim_param_from
        command += ' --alpha ' + str(alpha)
        command += ' --sim_param_layers ' + str(sim_param_layers)
        command += ' --sim_param_units ' + str(sim_param_units)
        command += ' --train_range_scale ' + str(train_range_scale)
        command += ' --num_sim_param_updates ' + str(num_sim_param_updates)
        if separate_trunks:
            command += ' --separate_trunks'
        if prop_train_range_scale:
            command += ' --prop_train_range_scale'
    if mean_scale is not None:
        command += ' --mean_scale ' + str(mean_scale)
    if run_type is not 'oracle':
        command += ' --dr'
        command += ' --range_scale ' + str(range_scale)
        if prop_range_scale:
            command += ' --prop_range_scale'
        if scale_large_and_small:
            command += ' --scale_large_and_small'

    # Defaults specific to the env
    if action_repeat is None:
        if domain_name == 'kitchen':
            action_repeat = 1
            command += ' --time_limit ' + str(time_limit)
            command += ' --delay_steps ' + str(delay_steps)
        elif task_name in ['walk', 'spin']:
            action_repeat = 2
        elif task_name in ['run', 'catch']:
            action_repeat = 4
    command += ' --action_repeat ' + str(action_repeat)

    if dr_option is None:
        if domain_name == 'kitchen':
            dr_option = 'nonconflicting_dr'
        elif 'dmc' in domain_name:
            dr_option = 'simple_dr'
    if not run_type == 'oracle':
        command += ' --dr_option ' + dr_option

    print(command)


# DIFFERENCES
# - Both
# - Model size
# - range_scale/train_range_scale
