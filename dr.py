import numpy as np

def config_dr(config):
    if 'dmc' in config.domain_name:
        config = config_dr_dmc(config)
    elif 'metaworld' in config.domain_name:
        config = config_dr_metaworld(config)
    elif 'kitchen' in config.domain_name:
        config = config_dr_kitchen(config)
    elif 'dummy' in config.domain_name:
        config = config_dummy(config)
    else:
        config.dr = {}
        config.real_dr_params = {}
        config.dr_list = []

    for k, v in config.dr.items():
        print(k)
        print(v)

    if config.mean_only:
        config.initial_dr_mean = np.array([config.dr[param] for param in config.real_dr_list])
    else:
        config.initial_dr_mean = np.array([config.dr[param][0] for param in config.real_dr_list])
        config.initial_dr_range = np.array([config.dr[param][1] for param in config.real_dr_list])
    return config

def config_dr_dmc(config):
    # Define constants from the dmc textures:
    # https://github.com/deepmind/dm_control/blob/33cea512329bdea46db177dd6c2f8ccb7b68ae73/dm_control/suite/common/materials.xml
    # I'm least certain about this one.  I'm not sure why we have one set of rgb colors for the whole ground,
    # but it seems to work.
    ground_r = 1.
    ground_g = 1.
    ground_b = 1.

    # "self" texture, used for dmc agents
    self_r = .7
    self_g = .5
    self_b = .3

    # "effector" texture, used for end-effectors
    effector_r = .7
    effector_g = .4
    effector_b = .2

    dr_option = config.dr_option
    if "ball_in_cup" in config.domain_name:
        real_dr_values = {
            "cup_mass": .06247,  # Significant changes
            "ball_mass": .06545,  # Significant changes
            "cup_damping": 3.,  # Significant changes
            "ball_damping": 0.,  # Negligible changes
            "actuator_gain": 1.,  # Significant changes
            "cup_r": self_r,
            "cup_g": self_g,
            "cup_b": self_b,
            "ball_r": effector_r,
            "ball_g": effector_g,
            "ball_b": effector_b,
            "ground_r": ground_r,
            "ground_g": ground_g,
            "ground_b": ground_b,
        }
        if dr_option == 'all_dr':
            config.real_dr_list = list(real_dr_values.keys())
        elif dr_option == 'nonconflicting_dr':
            config.real_dr_list = [
                "cup_mass", "ball_mass", "cup_r", "cup_g", "cup_b", "ball_r", "ball_g", "ball_b", "ground_r",
                "ground_g", "ground_b",
            ]
        elif dr_option == 'visual_dr':
            config.real_dr_list = [
                "cup_r", "cup_g", "cup_b", "ball_r", "ball_g", "ball_b",
            ]
        elif dr_option == 'mass_dr':
            config.real_dr_list = [
                'ball_mass'
            ]
        elif dr_option == 'cup_mass_dr':
            config.real_dr_list = [
                'cup_mass'
            ]
        elif dr_option == 'ball_mass_dr':
            config.real_dr_list = [
                'ball_mass'
            ]
        elif dr_option == 'simple_dr':
            config.real_dr_list = [
                "ball_mass", "cup_r", "cup_g", "cup_b", "ball_r", "ball_g", "ball_b", "ground_r", "ground_g", "ground_b",
            ]
    elif "walker" in config.domain_name:
        real_dr_values = {
            "torso_mass": 10.3138485,  # significant difference
            "right_thigh_mass": 3.9269907,
            "right_leg_mass": 2.7143362,
            "right_foot_mass": 1.9634954,
            "left_thigh_mass": 3.9269907,  # negligible difference
            "left_leg_mass": 2.7143362,  # small difference
            "left_foot_mass": 1.9634954,  # small difference
            "right_hip": .1,
            "right_knee": .1,
            "right_ankle": .1,
            "left_hip": .1,  # negligible difference
            "left_knee": .1,  # negligible difference
            "left_ankle": .1,  # negligible difference
            "ground_r": ground_r,
            "ground_g": ground_g,
            "ground_b": ground_b,
            "body_r": self_r,
            "body_g": self_g,
            "body_b": self_b,
            "full_mass": 10.3138485,
        }
        if dr_option == 'all_dr':
            config.real_dr_list = list(real_dr_values.keys())
        elif dr_option == 'nonconflicting_dr':
            config.real_dr_list = [
                "torso_mass", "right_thigh_mass", "right_leg_mass", "right_foot_mass", "left_thigh_mass",
                "left_leg_mass", "left_foot_mass", "ground_r", "ground_g", "ground_b", "body_r", "body_g", "body_b"
            ]
        elif dr_option == 'simplified_randomization':
            config.real_dr_list = [
                "full_mass", "left_foot_mass", "ground_r", "ground_g", "ground_b", "body_r", "body_g", "body_b"
            ]
        elif dr_option == 'visual_dr':
            config.real_dr_list = [
                "ground_r", "ground_g", "ground_b", "body_r", "body_g", "body_b"
            ]
        elif dr_option == 'minimass_dr':
            config.real_dr_list = [
                "left_foot_mass"
            ]
        elif dr_option == 'mass_dr':
            config.real_dr_list = [
                'torso_mass'
            ]
        elif dr_option == 'single_visual_dr':
            config.real_dr_list = [
                "ground_r"
            ]
        elif dr_option == 'simple_dr':
            config.real_dr_list = [
                "full_mass", "ground_r", "ground_g", "ground_b", "body_r", "body_g", "body_b"
            ]
    elif "finger" in config.domain_name:
        real_dr_values = {
            "proximal_mass": .805,  # no difference whatsoever
            "distal_mass": .636,  # minor differnce
            "spinner_mass": 2.32,  # significant difference
            "proximal_damping": 2.5,  # significant difference
            "distal_damping": 2.5,  # significant difference
            "hinge_damping": .5,  # no difference whatsoever
            "ground_r": ground_r,
            "ground_g": ground_g,
            "ground_b": ground_b,
            "finger_r": self_r,
            "finger_g": self_g,
            "finger_b": self_b,
            "hotdog_r": self_r,
            "hotdog_g": self_g,
            "hotdog_b": self_b,
        }
        if dr_option == 'all_dr':
            config.real_dr_list = list(real_dr_values.keys())
        elif dr_option == 'nonconflicting_dr':
            config.real_dr_list = [
                "proximal_damping", "distal_damping", "spinner_mass", "ground_r", "ground_g", "ground_b", "finger_r",
                "finger_g", "finger_b", "hotdog_r", "hotdog_g", "hotdog_b",
            ]
        elif dr_option == 'simple_dr':
            config.real_dr_list = [
                "proximal_damping", "spinner_mass", "ground_r", "ground_g", "ground_b", "finger_r",
                "finger_g", "finger_b", "hotdog_r", "hotdog_g", "hotdog_b",
            ]
    elif "cheetah" in config.domain_name:
        real_dr_values = {
            "torso_mass": 6.3603134,  # decent difference
            "bthigh_mass": 1.535248,  # decent difference
            "bshin_mass": 1.5809399,  # decent difference
            "bfoot_mass": 1.0691906,  # decent difference
            "fthigh_mass": 1.4255874,  # decent difference
            "fshin_mass": 1.1788511,  # decent difference
            "ffoot_mass": 0.84986943,  # decent difference
            "bthigh_damping": 6.,  # minor difference
            "bshin_damping": 4.5,  # minor difference
            "bfoot_damping": 3.,  # minor difference
            "fthigh_damping": 4.5,  # minor difference
            "fshin_damping": 3.,  # negligible difference
            "ffoot_damping": 1.5,  # negligible difference
            "ground_r": ground_r,
            "ground_g": ground_g,
            "ground_b": ground_b,
            "body_r": self_r,
            "body_g": self_g,
            "body_b": self_b,
            "full_mass": 6.3603134,
        }
        if dr_option == 'all_dr':
            config.real_dr_list = list(real_dr_values.keys())
        elif dr_option == 'nonconflicting_dr':
            config.real_dr_list = [
                "torso_mass", "ffoot_mass", "bfoot_mass", "ground_r", "ground_g", "ground_b", "body_r",
                "body_g", "body_b",
            ]
        elif dr_option == 'simple_dr':
            config.real_dr_list = [
                "full_mass", "ground_r", "ground_g", "ground_b", "body_r",
                "body_g", "body_b",
            ]
    config.real_dr_params = real_dr_values

    if config.simple_randomization:
        if "cup_catch" in config.task_name:
            config.real_dr_list = ["ball_mass"]
        elif "walker" in config.task_name:
            config.real_dr_list = ["torso_mass"]
        elif "finger" in config.task_name:
            config.real_dr_list = ["distal_mass"]
        elif "cheetah" in config.task_name:
            config.real_dr_list = ["torso_mass"]
    mean_scale = config.mean_scale
    range_scale = config.range_scale
    config.dr = {}  # (mean, range)
    # We store the random state and re-set it later so that the randomization is consistent, but other than that we
    # use the random seed passed into the model.
    temp = []
    rng_state = np.random.get_state()
    np.random.seed(0)
    for key in config.real_dr_list:
        cur_scale = mean_scale
        if config.scale_large_and_small:
            if bool(np.random.choice(a=[False, True], size=(1,))):
                cur_scale = 1/cur_scale
                temp.append(key)

        real_val = config.real_dr_params[key]
        if real_val == 0:
            real_val = 5e-2
        mean_val = real_val * cur_scale
        # Clip all visual values except dmc ground, which uses a different color scaling scheme, presumably b/c it's
        # a pattern not a single color.
        if key[-2:] in ['_r', '_g', '_b'] and not key in ['ground_r', 'ground_g', 'ground_b']:
            mean_val = min(real_val * cur_scale, 1.)
        if config.mean_only:
            config.dr[key] = mean_val
        else:
            config.dr[key] = (mean_val, real_val * range_scale)
    np.random.set_state(rng_state)
    config.sim_params_size = len(config.real_dr_list)
    return config


def config_dr_kitchen(config):
    dr_option = config.dr_option
    if config.simple_randomization:
        if 'rope' in config.task_name:
            config.real_dr_params = {
                "cylinder_mass": .5
            }
            config.dr = {  # (mean, range)
                "cylinder_mass": (config.mass_mean, config.mass_range)
            }
            config.real_dr_list = ["cylinder_mass"]
            config.sim_params_size = 1
        elif 'open_microwave' in config.task_name:
            config.real_dr_params = {
                "microwave_mass": .26
            }
            config.dr = {  # (mean, range)
                "microwave_mass": (config.mass_mean, config.mass_range)
            }
            config.real_dr_list = ['microwave_mass']
            config.sim_params_size = 1
        elif 'open_cabinet' in config.task_name:
            config.real_dr_params = {
                "cabinet_mass": 3.4
            }
            config.dr = {  # (mean, range)
                "cabinet_mass": (config.mass_mean, config.mass_range)
            }
            config.real_dr_list = ['cabinet_mass']
            config.sim_params_size = 1
        else:
            config.real_dr_params = {
                "kettle_mass": 1.08
            }
            config.dr = {  # (mean, range)
                "kettle_mass": (config.mass_mean, config.mass_range)
            }
            config.real_dr_list = ['kettle_mass']
            config.sim_params_size = 1
    else:
        if 'rope' in config.task_name:
            config.real_dr_params = {
                "joint1_damping": 10,
                "joint2_damping": 10,
                "joint3_damping": 5,
                "joint4_damping": 5,
                "joint5_damping": 5,
                "joint6_damping": 2,
                "joint7_damping": 2,
                "robot_b": 0.95,
                "robot_g": 0.95,
                "robot_r": 0.95,
                "cylinder_b": 0.1,
                "cylinder_g": .8,
                "cylinder_r": 1.,
                "cylinder_mass": 0.5,
                "box1_r": .75,
                "box1_g": .75,
                "box1_b": .75,
                "box2_r": .75,
                "box2_g": .75,
                "box2_b": .75,
                "box3_r": .75,
                "box3_g": .75,
                "box3_b": .75,
                "box4_r": .75,
                "box4_g": .75,
                "box4_b": .75,
                "box5_r": .75,
                "box5_g": .75,
                "box5_b": .75,
                # "box6_r": .2,
                # "box6_g": 1,
                # "box6_b": .2,
                # "box7_r": .2,
                # "box7_g": 1,
                # "box7_b": .2,
                # "box8_r": .2,
                # "box8_g": 1,
                # "box8_b": .2,
                "rope_len_max" : 0.12,
                "rope_damping": 0,
                "rope_friction": 0,
                "rope_stiffness": 0,
                "lighting": 0.3
            }
            if dr_option == 'partial_dr':
                config.real_dr_list = ["cylinder_mass", "rope_damping", "rope_friction", "rope_stiffness"]
            elif dr_option == 'all_dr':
                config.real_dr_list = [
                    "joint1_damping", "joint2_damping", "joint3_damping", "joint4_damping", "joint5_damping",
                    "joint6_damping",
                    "joint7_damping", "robot_b", "robot_g", "robot_r", "cylinder_b", "cylinder_g",
                    "cylinder_r", "cylinder_mass", "box1_r", "box1_g", "box1_b", "box2_r", "box2_g", "box2_b", "box3_r",
                    "box3_g", "box3_b", "box4_r", "box4_g", "box4_b", "box5_r", "box5_g", "box5_b",
                    # "box6_r", "box6_g",  "box6_b", "box7_r", "box7_g", "box7_b", "box8_r", "box8_g", "box8_b",
                    "rope_damping", "rope_friction",
                    "rope_stiffness", "lighting",
                ]
            elif dr_option == 'nonconflicting_dr':
                config.real_dr_list = [
                    "cylinder_b", "cylinder_g",
                    "cylinder_r", "cylinder_mass", "box1_r", "box1_g", "box1_b",
                    #"box2_r", "box2_g", "box2_b", "box3_r", "box3_g", "box3_b", "box4_r", "box4_g", "box4_b", "box5_r", "box5_g", "box5_b",
                    # "box6_r", "box6_g", "box6_b", "box7_r", "box7_g", "box7_b", "box8_r", "box8_g", "box8_b",
                    "rope_damping", "lighting",
                ]
            elif dr_option == 'len_vis_dr':
                config.real_dr_list = [
                    "cylinder_b", "cylinder_g", "cylinder_r", "rope_len_max", "box1_r", "box1_g", "box1_b",  "lighting",
                ]
            elif dr_option == 'mass_vis_dr':
                config.real_dr_list = [
                    "cylinder_b", "cylinder_g", "cylinder_r", "cylinder_mass", "box1_r", "box1_g", "box1_b",  "lighting",
                ]
            elif dr_option == 'len_mass_vis_dr':
                config.real_dr_list = [
                    "cylinder_b", "cylinder_g", "cylinder_r", "cylinder_mass", "box1_r", "box1_g", "box1_b",  "lighting", "rope_len_max",
                ]

        elif 'real_p' in config.task_name:
            config.real_dr_params = {
                'box_mass': 0.023013,
                'box_r': 0.3,
                'box_g': 0.5,
                'box_b': 1,
                "table_b": 1,
                "table_g": 1,
                "table_r": 1,
                "table_friction": 1,
            }
            if dr_option == 'all_dr':
                config.real_dr_list = list(config.real_dr_params.keys())
            elif dr_option == 'nonconflicting_dr':
                config.real_dr_list = ['box_mass', 'box_r', 'box_g', 'box_b', 'table_b', 'table_g', 'table_r']
            elif dr_option == 'box_mass':
                config.real_dr_list = ['box_mass']
            elif dr_option == 'visual_dr':
                config.real_dr_list = ['box_r', 'box_g', 'box_b', 'table_r', 'table_g', 'table_b']
        elif 'real_c' in config.task_name:
            config.real_dr_params = {
                'cabinet_mass': .34,
                'cabinet_friction': 5,
                'cabinet_r': .46,
                'cabinet_g': .5,
                'cabinet_b': .6,
                'cabinet_handle_r': 1,
                'cabinet_handle_g': 1,
                'cabinet_handle_b': 1,
                "table_b": 1,
                "table_g": 1,
                "table_r": 1,
            }
            if dr_option == 'all_dr':
                config.real_dr_list = list(config.real_dr_params.keys())
            elif dr_option == 'nonconflicting_dr':
                config.real_dr_list = ['cabinet_friction', 'cabinet_r', 'cabinet_g', 'cabinet_b', 'table_b',
                                       'table_g', 'table_r']
            elif dr_option == 'dynamics_dr':
                config.real_dr_list = ['cabinet_mass']
            elif dr_option == 'visual_dr':
                config.real_dr_list = ['cabinet_r', 'cabinet_g', 'cabinet_b', 'cabinet_handle_r', 'cabinet_handle_g',
                                       'cabinet_handle_b', 'table_r', 'table_g', 'table_b']
        else:
            config.real_dr_params = {
                "cabinet_b": 0.5,  # OK
                "cabinet_friction": 1,  # hard 2 tell
                "cabinet_g": 0.5,  # OK
                "cabinet_mass": .34,  # hard 2 tell
                "cabinet_r": 0.5,  # OK
                "joint1_damping": 10,  # negligible
                "joint2_damping": 10,  # negligible
                "joint3_damping": 5,  # negligible
                "joint4_damping": 5,  # negligible
                "joint5_damping": 5,  # negligible
                "joint6_damping": 2,  # negligible
                "joint7_damping": 2,  # negligible
                "kettle_b": 0.5,  # OK
                "kettle_friction": 1.0,  # No change
                "kettle_g": 0.5,  # OK
                "kettle_mass": 1.08,  # Making mass SMALLEER make a minor difference, making mass bigger does not
                "kettle_r": 0.5,  # OK
                "knob_mass": 0.02,  # unlearnable
                "lighting": 0.3,  # great
                "microwave_b": 0.5,  # OK
                "microwave_friction": 1,  # probably unlearnable
                "microwave_g": 0.5,  # OK
                "microwave_mass": .26,  # probably unlearnable
                "microwave_r": 0.5,  # OK
                "robot_b": 0.92,  # OK
                "robot_g": .99,  # OK
                "robot_r": 0.95,  # OK
                "stove_b": 0.5,  # OK
                "stove_friction": 1.,  # minor change
                "stove_g": 0.5,  # OK
                "stove_r": 0.5,  # OK
            }
            if dr_option == 'test_dr':
                config.real_dr_list = [
                    "lighting", "kettle_mass", "kettle_friction", "stove_friction",
                ]
            elif dr_option == 'partial_dr':
                config.real_dr_list = [
                    "cabinet_b", "cabinet_g", "cabinet_mass", "cabinet_r", "joint7_damping", "kettle_b",
                    "kettle_g", "kettle_mass", "kettle_r", "lighting", "microwave_b", "kettle_friction",
                    "microwave_g", "microwave_mass", "microwave_r", "robot_b", "robot_g", "robot_r", "stove_b",
                    "stove_g", "stove_r",
                ]
            elif dr_option == 'all_dr':
                config.real_dr_list = [
                    "cabinet_b", "cabinet_friction", "cabinet_g", "cabinet_mass", "cabinet_r", "joint1_damping",
                    "joint2_damping",
                    "joint3_damping", "joint4_damping", "joint5_damping", "joint6_damping", "joint7_damping",
                    "kettle_b", "kettle_friction",
                    "kettle_g", "kettle_mass", "kettle_r", "knob_mass", "lighting", "microwave_b", "microwave_friction",
                    "microwave_g", "microwave_mass", "microwave_r", "robot_b", "robot_g", "robot_r", "stove_b",
                    "stove_friction", "stove_g", "stove_r",
                ]
            elif dr_option == 'dynamics_dr':
                config.real_dr_list = ["cabinet_mass", "joint7_damping", "kettle_mass", "kettle_friction"]
            elif dr_option == 'friction_dr':
                config.real_dr_list = ["kettle_friction", "cabinet_friction"]
            elif dr_option == 'dynamics_nonconflicting_dr':
                config.real_dr_list = ["cabinet_mass", "joint7_damping", "kettle_mass"]
            elif dr_option == 'nonconflicting_dr':
                # would be cool if we could get stove_friction in here, but I think it interacts
                config.real_dr_list = ["cabinet_mass", "kettle_mass", "cabinet_b", "cabinet_g",
                                       "cabinet_r",
                                       "kettle_b", "kettle_g", "kettle_r", "lighting", "microwave_b", "microwave_g",
                                       "microwave_r", "robot_b", "robot_g", "robot_r", "stove_b", "stove_g", "stove_r"]
            elif dr_option == 'visual':
                config.real_dr_list = ["stove_r"]
            elif dr_option == 'mass':
                config.real_dr_list = ["kettle_mass" if "kettle" in config.task else "cabinet_mass"]
            elif dr_option == 'friction':
                config.real_dr_list = ["kettle_friction" if "kettle" in config.task else "cabinet_friction"]

            if 'slide' in config.task_name:
                config.real_dr_params['stove_friction'] = 1e-3
                config.real_dr_params['kettle_friction'] = 1e-3

            # Remove kettle-related d-r for the microwave task, which has no kettle present.
            if 'open_microwave' in config.task_name:
                for k in list(config.real_dr_params.keys()):
                    if 'kettle' in k:
                        del config.real_dr_params[k]

        config.sim_params_size = len(config.real_dr_list)
        mean_scale = config.mean_scale
        range_scale = config.range_scale
        config.dr = {}  # (mean, range)

        rng_state = np.random.get_state()
        np.random.seed(0)

        for key in config.real_dr_list:
            real_val = config.real_dr_params[key]
            cur_scale = mean_scale
            if config.scale_large_and_small:
                if bool(np.random.choice(a=[False, True], size=(1,))):
                    cur_scale = 1 / cur_scale
            if real_val == 0:
                real_val = 5e-2
            if '_r' in key or '_g' in key or '_b' in key and not key in ['cabinet_handle_r', 'cabinet_handle_g', 'cabinet_handle_b']:
                config.dr[key] = (min(real_val * cur_scale, 1.), real_val * range_scale) #clip to possible colour value
            else:
                config.dr[key] = (real_val * cur_scale, real_val * range_scale)

        np.random.set_state(rng_state)

            # Keep mean only
    if config.mean_only and config.dr is not None:
        dr = {}
        for key, vals in config.dr.items():
            dr[key] = vals[0]  # only keep mean
        config.dr = dr
    return config


def config_dr_metaworld(config):
  if config.dr_option == 'simple':
      if 'basketball' in config.task_name:
        config.real_dr_params = {
          "object_mass": .01
        }
        config.dr = {  # (mean, range)
          "object_mass": (config.mass_mean, config.mass_range)
        }
        config.real_dr_list = ["object_mass"]
        config.sim_params_size = 1
      elif 'stick' in config.task_name or 'basketball' in config.task_name:
          config.real_dr_params = {
            "object_mass": .128  # TODO: ???
          }
          config.dr = {  # (mean, range)
            "object_mass": (config.mass_mean, config.mass_range)
          }
          config.real_dr_list = ["object_mass"]
          config.sim_params_size = 1
  elif config.dr_option == 'all_dr':
      real_dr_joint = {
        "table_friction": 2.,
        "table_r": .6,
        "table_g": .6,
        "table_b": .5,
        "robot_friction": 1.,
        "robot_r": .5,
        "robot_g": .1,
        "robot_b": .1,
      }
      if 'basketball' in config.task_name:
        config.real_dr_params = {
          "basket_friction": .5,
          "basket_goal_r": .5,
          "basket_goal_g": .5,
          "basket_goal_b": .5,
          "backboard_r": .5,
          "backboard_g": .5,
          "backboard_b": .5,
          "object_mass": .01,
          "object_friction": 1.,
          "object_r": 0.,
          "object_g": 0.,
          "object_b": 0.,
        }
        config.real_dr_params.update(real_dr_joint)
        config.real_dr_list = list(config.real_dr_params.keys())
      elif 'stick' in config.task_name:
        config.real_dr_params = {
          "stick_mass": 1.,
          "stick_friction": 1.,
          "stick_r": 1.,
          "stick_g": .3,
          "stick_b": .3,
          "object_mass": .128,
          "object_friction": 1.,
          "object_body_r": 0.,
          "object_body_g": 0.,
          "object_body_b": 1.,
          "object_handle_r": 0,
          "object_handle_g": 0,
          "object_handle_b": 0,
        }
        config.real_dr_params.update(real_dr_joint)
        config.real_dr_list = list(config.real_dr_params.keys())
      else:
        config.real_dr_params = real_dr_joint
        config.real_dr_list = list(config.real_dr_params.keys())
      config.sim_params_size = len(config.real_dr_list)
      mean_scale = config.mean_scale
      range_scale = config.range_scale
      config.dr = {}  # (mean, range)

      rng_state = np.random.get_state()
      np.random.seed(0)

      for key in config.real_dr_list:
        real_val = config.real_dr_params[key]
        cur_scale = mean_scale
        if config.scale_large_and_small:
            if bool(np.random.choice(a=[False, True], size=(1,))):
                cur_scale = 1/cur_scale
        if real_val == 0:
          real_val = 5e-2
        if key[-2:] in ['_r', '_g', '_b']:
            mean_val = min(real_val * cur_scale, 1.)
        if config.mean_only:
            config.dr[key] = mean_val
        else:
            config.dr[key] = (mean_val, real_val * range_scale)
      np.random.set_state(rng_state)
  return config

def config_dummy(config):
    real_dr_values = {
        "square_size": 6.,
        "speed_multiplier": 5.,
        "square_r": .5,
        "square_g": .5,
        "square_b": 0.,
    }
    dr_option = config.dr_option
    if dr_option == 'all_dr':
        config.real_dr_list = list(real_dr_values.keys())
    elif dr_option == 'red':
        config.real_dr_list = ['square_r']
    elif dr_option == 'visual_dr':
        config.real_dr_list = ['square_r', 'square_g', 'square_b']
    elif dr_option == 'speed':
        config.real_dr_list = ['speed_multiplier']
    elif dr_option == 'size':
        config.real_dr_list = ['square_size']
    config.real_dr_params = real_dr_values
    cur_scale = config.mean_scale
    range_scale = config.range_scale
    config.dr = {}  # (mean, range)

    for key, real_val in config.real_dr_params.items():
        if config.scale_large_and_small:
            cur_scale = 1 / cur_scale #alternate scaling

        if real_val == 0:
            real_val = 5e-2
        if key[-2:] in ['_r', '_g', '_b']:
            mean_val = min(real_val * cur_scale, 1.)
        if config.mean_only:
            config.dr[key] = mean_val
        else:
            config.dr[key] = (mean_val, real_val * range_scale)

    config.sim_params_size = len(config.real_dr_list)
    return config