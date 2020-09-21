# CURL: Contrastive Unsupervised Representation Learning for Sample-Efficient Reinforcement Learning

This repository is the official implementation of [CURL](https://mishalaskin.github.io/curl/) for the DeepMind control experiments. Atari experiments were done in a separate codebase that will also be made publicly available shortly. Our implementation of SAC is based on [SAC+AE](https://github.com/denisyarats/pytorch_sac_ae) by Denis Yarats. 

## Installation 

All of the dependencies are in the `conda_env.yml` file. They can be installed manually or with the following command:

```
conda env create -f conda_env.yml
```

Enter the environments directory and run
```
pip install -e .
```


## Instructions
To train a CURL agent on the `cartpole swingup` task from image-based observations run `bash script/run.sh` from the root of this directory. The `run.sh` file contains the following command, which you can modify to try different environments / hyperparamters.
```
CUDA_VISIBLE_DEVICES=0 python train.py \
    --domain_name cartpole \
    --task_name swingup \
    --encoder_type pixel \
    --action_repeat 8 \
    --save_tb --pre_transform_image_size 100 --image_size 84 \
    --work_dir ./tmp \
    --agent curl_sac --frame_stack 3 \
    --seed -1 --critic_lr 1e-3 --actor_lr 1e-3 --eval_freq 10000 --batch_size 128 --num_train_steps 1000000 
```

In your console, you should see printouts that look like:

```
| train | E: 221 | S: 28000 | D: 18.1 s | R: 785.2634 | BR: 3.8815 | A_LOSS: -305.7328 | CR_LOSS: 190.9854 | CU_LOSS: 0.0000
| train | E: 225 | S: 28500 | D: 18.6 s | R: 832.4937 | BR: 3.9644 | A_LOSS: -308.7789 | CR_LOSS: 126.0638 | CU_LOSS: 0.0000
| train | E: 229 | S: 29000 | D: 18.8 s | R: 683.6702 | BR: 3.7384 | A_LOSS: -311.3941 | CR_LOSS: 140.2573 | CU_LOSS: 0.0000
| train | E: 233 | S: 29500 | D: 19.6 s | R: 838.0947 | BR: 3.7254 | A_LOSS: -316.9415 | CR_LOSS: 136.5304 | CU_LOSS: 0.0000
```

For reference, the maximum score for cartpole swing up is around 845 pts, so CURL has converged to the optimal score. This takes about and hour of training depending on your GPU. 

Log abbreviation mapping:

```
train - training episode
E - total number of episodes 
S - total number of environment steps
D - duration in seconds to train 1 episode
R - mean episode reward
BR - average reward of sampled batch
A_LOSS - average loss of actor
CR_LOSS - average loss of critic
CU_LOSS - average loss of the CURL encoder
```

All data related to the run is stored in the specified `working_dir`. To enable model or video saving, use the `--save_model` or `--save_video` flags. For all available flags, inspect `train.py`. To visualize progress with tensorboard run:

```
tensorboard --logdir log --port 6006
```

and go to `localhost:6006` in your browser. If you're running headlessly, try port forwarding with ssh. 

For GPU accelerated rendering, make sure EGL is installed on your machine and set `export MUJOCO_GL=egl`. For environment troubleshooting issues, see the DeepMind control documentation.


# Debugging common installation errors
Error message `ERROR: GLEW initalization error: Missing GL version`
- Make sure `/usr/lib/x86_64-linux-gnu/libGLEW.so` and `/usr/lib/x86_64-linux-gnu/libGL.so` exist.  If not, apt-install them.
- Try trying adding the powerset of those two paths to LD_PRELOAD.

Error `Shadow framebuffer is not complete, error 0x8cd7`
- Like above, make sure libglew and libGL are installed.
- If `/usr/lib/nvidia` exists but '/usr/lib/nvidia-430/` (or some other number) does not exist, run `ln -s /usr/lib/nvidia /usr/lib/nvidia-430`.  It may have to match the number of your nvidia driver, I'm not sure.
- Consider adding that symlink to LD_LIBRARY PATH.
- If /usr/lib/nvidia doesn't exist, and neither does /usr/lib/nvidia-xxx, then create the folder `/usr/lib/nvidia /usr/lib/nvidia-430`.

Error message `RuntimeError: Failed to initialize OpenGL: 
- Make sure MUJOCO_GL is correct (egl for DMC, osmesa for anything else).

Which error is the best?  Framebuffer > GLEW initialization > Failed to initialize OpenGL.
