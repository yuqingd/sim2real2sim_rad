import argparse


def generate_shell_commands(domain_name, task_name, run_type, save=False, action_repeat=None, alpha=.1,
                            seed=1, eval_freq=2000, scale_large_and_small=True, delay_steps=2, time_limit=60,
                            mean_scale=None, prop_range_scale=True, prop_train_range_scale=True,
                            separate_trunks=True, train_range_scale=1, range_scale=None,
                            dr_option=None, update_sim_param_from='buffer', num_eval_episodes=5,
                            num_sim_param_updates=3, alternate=False, range_scale_sp=1, num_train_steps=None):
    # NOTE: the run_type BCS here is the sim_BCS (i.e. centered, small range), not the real_BCS (i.e. centered, big range)
    # To get the real_BCS, get a baseline run but pass in mean_scale 1.

    command = 'CUDA_VISIBLE_DEVICES=X python train.py --gpudevice X --id S'
    command += ' --domain_name ' + domain_name
    command += ' --task_name ' + task_name
    command += ' --seed ' + str(seed)
    command += ' --save_video'
    command += ' --save_tb'
    command += ' --use_img'
    command += ' --num_eval_episodes ' + str(num_eval_episodes)
    command += ' --eval_freq ' + str(eval_freq)

    if save:
        command += ' --save_model'
        command += ' --save_buffer'

    if alternate:
        command += ' --alternate_training'
        command += ' --range_scale_sp ' + str(range_scale_sp)
        if range_scale is None:
            range_scale = .1
    else:
        if range_scale is None:
            if run_type == 'BCS':
                range_scale = .1
            else:
                range_scale = 1

    # Defaults specific to the type of run (OL3, oracle, baseline)
    if run_type == 'oracle':
        command += ' --outer_loop_version 0'
    elif run_type == 'baseline':
        if mean_scale is None:
            mean_scale = 2
        command += ' --outer_loop_version 0'
    elif 'BCS' in run_type:
        command += ' --outer_loop_version 0'
        mean_scale = 1
    elif run_type == 'OL3':
        if mean_scale is None:
            mean_scale = 2
        command += ' --outer_loop_version 3'
        command += ' --prop_alpha'
        command += ' --update_sim_param_from ' + update_sim_param_from
        command += ' --alpha ' + str(alpha)
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
    if num_train_steps is None:
        if domain_name == 'kitchen':
            num_train_steps = 500000
        else:
            num_train_steps = int(1e6 / action_repeat)
    command += ' --num_train_steps ' + str(num_train_steps)

    if dr_option is None:
        if domain_name == 'kitchen':
            dr_option = 'nonconflicting_dr'
        elif 'dmc' in domain_name:
            dr_option = 'simple_dr'
    if not run_type == 'oracle':
        command += ' --dr_option ' + dr_option

    print(command)


if __name__ == '__main__':

    # GENERATE DMC COMMANDS
    envs = [
        ('dmc_ball_in_cup', 'catch'),
        ('dmc_walker', 'walk'),
        ('dmc_cheetah', 'run'),
        ('dmc_finger', 'spin')
    ]
    # ORACLE
    print("============ GENERATING ORACLE RUNS =============")
    for seed in range(3, 6):
        print(f"////// {seed} //////")
        for domain, task in envs:
            generate_shell_commands(domain, task, 'oracle', save=True, seed=seed)

    print("============ GENERATING BASELINE RUNS =============")
    for seed in range(3):
        print(f"////// {seed} //////")
        for domain, task in envs:
            generate_shell_commands(domain, task, 'baseline', save=True, seed=seed)

    print("============ GENERATING OL3 RUNS =============")
    for seed in range(3):
        print(f"////// {seed} //////")
        for domain, task in envs:
            generate_shell_commands(domain, task, 'OL3', save=True, seed=seed, alternate=True)

    print("============ GENERATING BCS RUNS =============")
    for domain, task in envs:
        generate_shell_commands(domain, task, 'BCS', save=True, seed=0)
