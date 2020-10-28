import argparse
import os
from shutil import copyfile
from pathlib import Path

directories = ['final_runs', 'logdir']
target_directory = 'tb_plots'


target_dir = Path(target_directory)
if not target_dir.exists():
    print("Making directory", target_dir.name)
    target_dir.mkdir()
else:
    print(f"Directory {target_dir.name} already exists; removing old files")
    for old_run in target_dir.iterdir():
        for old_file in old_run.iterdir():
            old_file.unlink()
        old_run.rmdir()

# Find relevant filenames
for directory in directories:
    directory = Path(directory)
    # Loop through files
    for run_file in directory.iterdir():
        print("Copying over", run_file.name)
        new_dir = target_dir.joinpath(run_file.name)
        new_dir.mkdir()

        # Copy over the tb file
        tb_file = run_file.joinpath('tb')
        events_file = list(tb_file.iterdir())[0]
        new_tb_file = new_dir.joinpath(events_file.name)
        copyfile(events_file, new_tb_file)

        # Copy over the reward/success file
        log_file = run_file.joinpath('eval_scores.npy')
        new_logfile = new_dir.joinpath('eval_scores.npy')
        copyfile(log_file, new_logfile)

        # Copy over the SPE files
        log_files = [f for f in run_file.iterdir() if 'agent-sim-params' in f.name]
        for log_file in log_files:
            new_logfile = new_dir.joinpath(log_file.name)
            copyfile(log_file, new_logfile)

print('scp -r tb_plots/ olivia@cthulhu3.ist.berkeley.edu:~/Sim2Real/Sim2/sim2real2sim_rad/paper_tb/')
