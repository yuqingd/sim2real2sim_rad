import argparse
import os
from shutil import copyfile

# Collect filename from input
parser = argparse.ArgumentParser(description="Download last video to Olivia's computer")
parser.add_argument('pattern', type=str, help='subset of the filename to match')
args = parser.parse_args()

def copy_last_video(file_name, prefix):
    videos = os.listdir(os.path.join('logdir', file_name, f'{prefix}_video'))
    videos = [os.path.join('logdir', file_name, f'{prefix}_video', v) for v in videos]
    videos.sort(key=os.path.getmtime, reverse=True)

    # Find the most recent video
    try:
        last_video = videos[0]
        video_name = os.path.basename(last_video)
        print("downloading ", prefix, video_name)
        download_file = os.path.join('download_files', file_name + video_name)
        copyfile(last_video, download_file)
    except Exception as e:
        print("Error downloading videos for ", prefix, file_name)
        print(e)

# Find relevant filenames
files = os.listdir('logdir')
files = [f for f in files if args.pattern in f]
print(f"found {len(files)} files")

# Loop through relevant filenames
for file_name in files:
    print("Finding videos for ", file_name)
    copy_last_video(file_name, "sim")
    copy_last_video(file_name, "real")

