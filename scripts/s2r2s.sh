#Baseline

CUDA_VISIBLE_DEVICES=3 python train.py --domain_name kitchen --action_repeat 1 --task_name rope --seed 1 \
--batch_size 128 --id S0266 --save_tb --save_video --eval_freq 1000 --num_train_steps 1000000 --num_eval_episodes 3 \
--gpudevice 0 --delay_steps 2 --dr_option all_dr --mean_scale 1 --dr --log_interval 5  --save_model --save_buffer

#Oracle

CUDA_VISIBLE_DEVICES=6 python train.py --domain_name kitchen --action_repeat 1 --task_name rope --seed 1 \
--batch_size 128 --id S0267 --save_tb --save_video --eval_freq 1000 --num_train_steps 1000000 --num_eval_episodes 3 \
--gpudevice 6 --delay_steps 2 --log_interval 5  --save_model --save_buffer

#Tuning
  python train.py --domain_name kitchen --action_repeat 1 --task_name rope --seed 0 --batch_size 128 --id S0268 --save_tb --save_video --eval_freq 5000 --num_train_steps 1000000 --num_eval_episodes 3 --gpudevice 0 --delay_steps 2 --dr_option all_dr --mean_scale 1 --dr --outer_loop_version 3  --save_model --save_buffer  --sim_param_lr 1e-3 --alpha 0.01 --log_interval 5 --train_sim_param_every 1