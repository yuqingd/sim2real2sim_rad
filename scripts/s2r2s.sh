#Baseline

CUDA_VISIBLE_DEVICES=3 python train.py --domain_name kitchen --action_repeat 1 --task_name rope --seed 1 \
--batch_size 128 --id S0266 --save_tb --save_video --eval_freq 1000 --num_train_steps 1000000 --num_eval_episodes 3 \
--gpudevice 0 --delay_steps 2 --dr_option all_dr --mean_scale 1 --dr --log_interval 5  --save_model --save_buffer

#Oracle

    python train.py --domain_name kitchen --action_repeat 1 --task_name rope --seed 0 --id S01003 --save_tb --save_video --eval_freq 2000 --num_train_steps 1000000 --delay_steps 2 --num_eval_episodes 2 --gpudevice 0 --dr_option nonconflicting_dr --mean_scale 1 --dr --outer_loop_version 0 --save_model --save_buffer  --range_scale .5  --use_img --encoder_type pixel --time_limit 60 --prop_range_scale --state_type robot

CUDA_VISIBLE_DEVICES=6 python train.py --domain_name kitchen --action_repeat 1 --task_name rope --seed 1 \
--batch_size 128 --id S0267 --save_tb --save_video --eval_freq 1000 --num_train_steps 1000000 --num_eval_episodes 3 \
--gpudevice 6 --delay_steps 2 --log_interval 5  --save_model --save_buffer

#Tuning
  python train.py --domain_name kitchen --action_repeat 1 --task_name rope --seed 0 --batch_size 128 --id debug_real --save_tb --save_video --eval_freq 2000 --num_train_steps 1000000 --num_eval_episodes 3 --gpudevice 0 --delay_steps 2 --dr_option all_dr --mean_scale 1 --dr --outer_loop_version 3  --save_model --save_buffer  --sim_param_lr 1e-3 --alpha 0.01 --log_interval 5 --train_sim_param_every 1 --separate_trunks True --use_img --train_range_scale 1


   python train.py --domain_name kitchen --action_repeat 1 --task_name rope --seed 0 --id pretrain_ol3 --save_tb --save_video --eval_freq 2000 --num_train_steps 100000 --delay_steps 2 --num_eval_episodes 2 --gpudevice 0 --dr_option all_dr --mean_scale 1 --dr --outer_loop_version 3 --save_model --save_buffer --sim_param_lr 1e-3 --alpha .1 --train_sim_param_every 1 --range_scale 1 --train_range_scale 5 --use_img --encoder_type pixel --num_sim_param_updates 5 --scale_large_and_small --prop_alpha --update_sim_param_from both --time_limit 60

    python train.py --domain_name kitchen --action_repeat 1 --task_name rope --seed 0 --id debug_len_ol3 --save_tb --save_video --eval_freq 2000 --num_train_steps 100000 --delay_steps 2 --num_eval_episodes 2 --gpudevice 0 --dr_option len_vis_dr --mean_scale 1 --dr --outer_loop_version 3 --save_model --save_buffer --sim_param_lr 1e-3 --alpha .1 --train_sim_param_every 1 --prop_range_scale --range_scale .5 --train_range_scale 5 --use_img --encoder_type pixel --num_sim_param_updates 5 --scale_large_and_small --prop_alpha --update_sim_param_from both --time_limit 60