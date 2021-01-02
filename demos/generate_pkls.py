import os, sys, pickle, glob


def chunkIt(n, num, offset=0):
    avg = n / float(num)
    out = []
    last = 0.0

    while last < n:
        out.append([int(last + offset), int(last + offset + avg)])
        last += avg

    return out


n_workers = 24

input_root = '/mnt/data/voxceleb/dev/mp4'
video_list = glob.glob(input_root + '/**/*.mp4', recursive=True)

begin_ends = chunkIt(len(video_list), n_workers, 0)

for i in range(n_workers):
    begin, end = begin_ends[i]
    video_list_i = video_list[begin:end]
    with open(f'video_list_{i}.pkl', 'wb') as f:
        pickle.dump(video_list_i, f)
