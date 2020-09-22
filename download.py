import argparse
import os

# import paramiko
# from scp import SCPClient
from shutil import copy

# Source: https://stackoverflow.com/questions/250283/how-to-scp-in-python
# def createSSHClient(server, port, user, password):
# # def createSSHClient(location):
#     client = paramiko.SSHClient()
#     client.load_system_host_keys()
#     client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
#     client.connect(server, port, user, password)
#     # client.connect(location)
#     return client

# Collect filename from input
parser = argparse.ArgumentParser(description="Download last video to Olivia's computer")
parser.add_argument('pattern', type=str, help='subset of the filename to match')
args = parser.parse_args()

# Find relevant filenames
files = os.listdir('logdir')
files = [f for f in files if args.pattern in f]
print(f"found {len(files)} files")

# Loop through relevant filenames
for file_name in files:
    print("Finding videos for ", file_name)
    videos = os.listdir(os.path.join('logdir', file_name, 'sim_video'))
    videos = [os.path.join('logdir', file_name, 'sim_video', v) for v in videos]
    videos.sort(key=os.path.getmtime, reverse=True)

    # Find the most recent video
    last_video = videos[0]
    print("downloading ", last_video)


    # Change of plans.  Since IDK an easy way to scp it down, I'm just gonna move it to a new directory and scp from there.
    directory = "download_files"
    copy(last_video, directory)

    # # Download it to my machine
    # ssh = createSSHClient(server, port, user, password)
    # scp = SCPClient(ssh.get_transport())
    # last_video_download_name = args.pattern + last_video[last_video.index('real'):]
    # scp.put(last_video, "~/Sim2Real/Sim2/" + last_video_download_name)


# TEST_metaworld-faucet-open-09-21-im84-b128-s0-curl_sac-pixel-crop



# Find the last video
# Scp down to my machine

