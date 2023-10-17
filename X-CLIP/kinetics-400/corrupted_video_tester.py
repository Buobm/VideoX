import subprocess
import decord
import time
from concurrent.futures import ThreadPoolExecutor

def is_video_corrupted(file_name):
    """
    Check if the video is corrupted using ffprobe.
    """
    file = file_name.split()[0]
    class_label = file_name.split()[1]
    file_path = "/home/mischa/ETH/semester_project/VideoX/X-CLIP/kinetics-400/k400/videos/" + file
    try:
        vr = decord.VideoReader(file_path)
        # num_frames_to_check = min(10, len(vr))  # check up to 10 frames or the entire video if shorter
        # for i in range(num_frames_to_check):
        #     frame = vr[i]
        return file,class_label, False, ''
    except Exception as e:
        return file, class_label, True, str(e)


def main():
    good_files = []
    files_to_check = []

    # Read file names from the file
    with open('/home/mischa/ETH/semester_project/VideoX/X-CLIP/kinetics-400/k400/annotations/train.txt', 'r') as f:
        for line in f:
            files_to_check.append(line)

    # Parallel processing of video files
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(is_video_corrupted, files_to_check))

    # Process results
    for file_name, class_label, corrupted, error_msg in results:
        if corrupted:
            print(f"File '{file_name}' is corrupted. Error: {error_msg}")
        else:
            good_files.append(file_name + ' ' + class_label)

    # Write good files to a new text file
    with open('/home/mischa/ETH/semester_project/VideoX/X-CLIP/kinetics-400/k400/annotations/good_test.txt', 'w') as f:
        for file_name in good_files:
            f.write(file_name + '\n')

    print("\nSummary:")
    if good_files:
        print(f"{len(good_files)} good files were written to 'good_files.txt'")
    else:
        print("No good files found!")

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nTotal execution time: {elapsed_time:.2f} seconds")