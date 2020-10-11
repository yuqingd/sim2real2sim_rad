from itertools import product
import pickle
import numpy as np

# Put these in for args like --dr where there is no value associated with the key.
PRESENT = 'present'
ABSENT = 'absent'


# If assign_gpus is true, you can specify which GPUs you have and how many slots are available on each.
# By default, it will fill up one gpu before going onto the next, which may or may not be what we want.
# If you don't want to assign gpus, make the gpus list empty.
starting_id = 999  # TODO: update this each time!
gpus = [(0, 2), (1, 1), (3, 1)]

# This is to grid search. If you don't want to grid search, manually write in the param_args list.

# Common; if we're confident we won't change these, we could move them to defaults
common_params = [
    [{"save_tb": [PRESENT]}],
    [{"save_video": [PRESENT]}],
    [{"save_model": [PRESENT]}],
    [{"save_buffer": [PRESENT]}],
    [{"use_img": [PRESENT]}],
    [{"encoder_type": ["pixel"]}],
    [{"num_eval_episodes": [5]}],
    [{"seed": [1]}],
]

# Params only useful for our method and the baseline (in oracle runs, these args are ignored)
dr_params = [
    [{"range_scale": [1]}],
    [{"scale_large_and_small": [PRESENT]}],
    [{"dr_option": ["all_dr"]}],
]

# Env-specific params
env_params = [
    [{"domain_name": ["dmc_walker"], "task_name": ["walk"], "action_repeat": [2]},
     {"domain_name": ["dmc_cheetah"], "task_name": ["run"], "action_repeat": [4]},
     {"domain_name": ["dmc_ball_in_cup"], "task_name": ["catch"], "action_repeat": [4]},
     {"domain_name": ["dmc_finger"], "task_name": ["spin"], "action_repeat": [2]},
     {"domain_name": ["kitchen"], "task_name": ["rope", "push_kettle_burner", "open_cabinet"], "action_repeat": [1]},
     ],
]

# Condition-specific params (oracle vs baseline vs OL3)
# Oracle
oracle_params = [
    [{"outer_loop_version": [0]}],
]

# BCS, Baseline
baseline_params = [
    [{"outer_loop_version": [0]}],
    [{"dr": [PRESENT]}],
    [{"mean_scale": [1, 2]}],
]

# Params only useful for our method (in other methods, these args are ignored)
ol3_params = [
    [{"outer_loop_version": [3]}],
    [{"dr": [PRESENT]}],
    [{"start_outer_loop": [5000]}],
    [{"train_sim_param_every": [1]}],
    [{"prop_alpha": [PRESENT]}],
    [{"update_sim_param_from": ['both']}],
    # commands below this comment are most likely to be changed
    [{"alpha": [.05]}],
    [{"mean_scale": [2]}],
    [{"train_range_scale": [5]}],
]

oracle_params = oracle_params + env_params + common_params
baseline_params = baseline_params + env_params + dr_params + common_params
ol3_params = ol3_params + env_params + dr_params + common_params

sweep_params_list = [
    # oracle_params,
    baseline_params,
    # ol3_params,
]

id = starting_id
total_num_commands = 0
args_dict = {}
command_index = 0

gpu_ids = []
for gpu_id, count in gpus:
    gpu_ids += [gpu_id] * count

for sweep_params in sweep_params_list:
    # Each param set has a group of parameters.  We will find each product of groups of parameters
    all_args = []
    for param_set1 in sweep_params:
        assert isinstance(param_set1, list)

        param_set1_args = []
        for param_set2 in param_set1:
            assert isinstance(param_set2, dict)

            # Find all combos same as we did before
            keys, values = zip(*param_set2.items())
            param_set2_product = [dict(zip(keys, vals)) for vals in product(*values)]
            param_set1_args += param_set2_product

        all_args.append(param_set1_args)

    all_product = list(product(*all_args))

    # Merge list of dictionaries into single one
    param_args = [{k: v for d in param_set0 for k, v in d.items()} for param_set0 in all_product]

    total_num_commands += len(param_args)


    command_strs = []
    arg_mapping_strs = []

    for args in param_args:
        full_id = f"S{id:04d}"
        args_command = ""
        args_mapping = f"ID: {full_id}; Parameters:"
        for k, v in args.items():
            args_mapping += f" {k}: {v},"
            if v == PRESENT:
                args_command += f" --{k}"
            elif v == ABSENT:
                continue
            else:
                args_command += f" --{k} {v}"

        gpu_index = gpu_ids[command_index] if command_index < len(gpu_ids) else "x"
        full_command = f'CUDA_VISIBLE_DEVICES={gpu_index} python train.py --gpudevice {gpu_index} --id {full_id} {args_command}'
        command_strs.append(full_command)
        arg_mapping_strs.append(args_mapping)
        args_dict[full_id] = args
        id += 1
        command_index += 1

    print("\n".join(command_strs))

print("TOTAL COMMANDS", total_num_commands)