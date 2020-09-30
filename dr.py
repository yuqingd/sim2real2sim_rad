import numpy as np

def config_dr(config):
    if 'dmc' in config.domain_name:
        config = config_dr_dmc(config)
    elif 'metaworld' in config.domain_name:
        config = config_dr_metaworld(config)
    elif 'kitchen' in config.domain_name:
        config = config_dr_kitchen(config)
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
    dr_option = config.dr_option
    if "ball_in_cup" in config.domain_name:
        real_dr_values = {
            "cup_mass": .0625,
            "ball_mass": .0654,
            "cup_damping": 3.,
            "ball_damping": 3.,
            "actuator_gain": 1.,
            "cup_r": .5,
            "cup_g": .5,
            "cup_b": .5,
            "ball_r": .5,
            "ball_g": .5,
            "ball_b": .5,
        }
        if dr_option == 'all_dr':
            config.real_dr_list = list(real_dr_values.keys())
        elif dr_option == 'nonconflicting_dr':
            config.real_dr_list = [
                "cup_mass", "ball_mass", "cup_r", "cup_g", "cup_b", "ball_r", "ball_g", "ball_b",
            ]
        elif dr_option == 'visual_dr':
            config.real_dr_list = [
                "cup_r", "cup_g", "cup_b", "ball_r", "ball_g", "ball_b",
            ]
        elif dr_option == 'mass_dr':
            config.real_dr_list = [
                'ball_mass'
            ]
    elif "walker" in config.domain_name:
        real_dr_values = {
            "torso_mass": 10.3,
            "right_thigh_mass": 3.93,
            "right_leg_mass": 2.71,
            "right_foot_mass": 1.96,
            "left_thigh_mass": 3.93,
            "left_leg_mass": 2.71,
            "left_foot_mass": 1.96,
            "right_hip": .1,
            "right_knee": .1,
            "right_ankle": .1,
            "left_hip": .1,
            "left_knee": .1,
            "left_ankle": .1,
            "ground_r": .5,
            "ground_g": .5,
            "ground_b": .5,
            "body_r": .5,
            "body_g": .5,
            "body_b": .5,
        }
        if dr_option == 'all_dr':
            config.real_dr_list = list(real_dr_values.keys())
        elif dr_option == 'nonconflicting_dr':
            config.real_dr_list = [
                "right_hip", "right_knee", "right_ankle", "left_hip", "left_knee", "left_ankle", "ground_r", "ground_g",
                "ground_b", "body_r", "body_g", "body_b"
            ]
        elif dr_option == 'visual_dr':
            config.real_dr_list = [
                "ground_r", "ground_g", "ground_b", "body_r", "body_g", "body_b"
            ]
        elif dr_option == 'mass_dr':
            config.real_dr_list = [
                'torso_mass'
            ]
    elif "finger" in config.domain_name:
        real_dr_values = {
            "proximal_mass": .805,
            "distal_mass": .636,
            "spinner_mass": 2.32,
            "proximal_damping": 2.5,
            "distal_damping": 2.5,
            "hinge_damping": .5,
            "ground_r": .5,
            "ground_g": .5,
            "ground_b": .5,
            "finger_r": .5,
            "finger_g": .5,
            "finger_b": .5,
            "hotdog_r": .5,
            "hotdog_g": .5,
            "hotdog_b": .5,
        }
        if dr_option == 'all_dr':
            config.real_dr_list = list(real_dr_values.keys())
        elif dr_option == 'nonconflicting_dr':
            config.real_dr_list = [
                "proximal_mass", "distal_mass", "spinner_mass", "ground_r", "ground_g", "ground_b", "finger_r",
                "finger_g",
                "finger_b", "hotdog_r", "hotdog_g", "hotdog_b",
            ]
    elif "cheetah" in config.domain_name:
        real_dr_values = {
            "torso_mass": 6.36,
            "bthigh_mass": 1.54,
            "bshin_mass": 1.58,
            "bfoot_mass": 1.07,
            "fthigh_mass": 1.43,
            "fshin_mass": 1.18,
            "ffoot_mass": .85,
            "bthigh_damping": 6,
            "bshin_damping": 4.5,
            "bfoot_damping": 3.,
            "fthigh_damping": 4.5,
            "fshin_damping": 3.,
            "ffoot_damping": 1.5,
            "ground_r": .5,
            "ground_g": .5,
            "ground_b": .5,
            "body_r": .5,
            "body_g": .5,
            "body_b": .5,
        }
        config.real_dr_list = list(config.real_dr_params.keys())
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
    for key in config.real_dr_list:
        real_val = config.real_dr_params[key]
        if real_val == 0:
            real_val = 5e-2
        if config.mean_only:
            config.dr[key] = real_val * mean_scale
        else:
            config.dr[key] = (real_val * mean_scale, real_val * range_scale)
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
                "cylinder_b": .2,
                "cylinder_g": .2,
                "cylinder_r": 1.,
                "cylinder_mass": 0.5,
                "box1_r": .2,
                "box1_g": 1,
                "box1_b": .2,
                "box2_r": .2,
                "box2_g": 1,
                "box2_b": .2,
                "box3_r": .2,
                "box3_g": 1,
                "box3_b": .2,
                "box4_r": .2,
                "box4_g": 1,
                "box4_b": .2,
                "box5_r": .2,
                "box5_g": 1,
                "box5_b": .2,
                # "box6_r": .2,
                # "box6_g": 1,
                # "box6_b": .2,
                # "box7_r": .2,
                # "box7_g": 1,
                # "box7_b": .2,
                # "box8_r": .2,
                # "box8_g": 1,
                # "box8_b": .2,
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
                    "joint7_damping", "robot_b", "robot_g", "robot_r", "cylinder_b", "cylinder_g",
                    "cylinder_r", "cylinder_mass", "box1_r", "box1_g", "box1_b", "box2_r", "box2_g", "box2_b", "box3_r",
                    "box3_g", "box3_b", "box4_r", "box4_g", "box4_b", "box5_r", "box5_g", "box5_b",
                    # "box6_r", "box6_g", "box6_b", "box7_r", "box7_g", "box7_b", "box8_r", "box8_g", "box8_b",
                    "rope_damping", "lighting",
                ]

        else:
            config.real_dr_params = {
                "cabinet_b": 0.5,
                "cabinet_friction": 1,
                "cabinet_g": 0.5,
                "cabinet_mass": 3.4,
                "cabinet_r": 0.5,
                "joint1_damping": 10,
                "joint2_damping": 10,
                "joint3_damping": 5,
                "joint4_damping": 5,
                "joint5_damping": 5,
                "joint6_damping": 2,
                "joint7_damping": 2,
                "kettle_b": 0.5,
                "kettle_friction": 1.0,
                "kettle_g": 0.5,
                "kettle_mass": 1.08,
                "kettle_r": 0.5,
                "knob_mass": 0.02,
                "lighting": 0.3,
                "microwave_b": 0.5,
                "microwave_friction": 1,
                "microwave_g": 0.5,
                "microwave_mass": .26,
                "microwave_r": 0.5,
                "robot_b": 0.92,
                "robot_g": .99,
                "robot_r": 0.95,
                "stove_b": 0.5,
                "stove_friction": 1.,
                "stove_g": 0.5,
                "stove_r": 0.5,
            }
            if dr_option == 'partial_dr':
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
                config.real_dr_list = ["cabinet_mass", "joint7_damping", "kettle_mass", "cabinet_b", "cabinet_g",
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
        for key in config.real_dr_list:
            real_val = config.real_dr_params[key]
            if real_val == 0:
                real_val = 5e-2
            config.dr[key] = (real_val * mean_scale, real_val * range_scale)

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
      for key in config.real_dr_list:
        real_val = config.real_dr_params[key]
        if real_val == 0:
          real_val = 5e-2
        if config.mean_only:
          config.dr[key] = real_val * mean_scale
        else:
          config.dr[key] = (real_val * mean_scale, real_val * range_scale)
  return config