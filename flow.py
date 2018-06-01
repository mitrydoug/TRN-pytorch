import os, sys

import cv2
import numpy as np

def compute_flow_for_video(video_fn):
    flows = []
    cap = cv2.VideoCapture(video_fn)
    ret, frame1 = cap.read()
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    ret, frame2 = cap.read()
    while ret:
        next = cvs.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
                prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flows.append(flow)
        ret, frame2 = cap.read()
        prvs = next
    # assume motion continues
    flows.append(flows[-1])
    accels = []
    for i, flow in enumerate(flows[:-1]):
        next = flows[i+1]
        accels.append(next-flow)
    accels.append(accels[-1])
    result = [np.concat([f, a], axis=2) for f, a in zip(flows, accels)]
    return result

root = None
backup = None

def process_split(split):
    split_file = f'{root}{split}Set.csv'
    frames_file = f'{root}{split}_frames.txt'

    with open(f'{root}moments_categories.txt', 'r') as f:
        labels = dict(tuple(line.strip().split(',')) for line in f)

    i = 0
    
    with open(split_file, 'r') as f:
        with open(frames_file, 'w') as g:
            for line in f:
                fn, label, _, _ = line.split(',')
                assert (fn[-4:] == '.mp4')
                frames = compute_flow_for_video(f'{root}{split}/{fn}')
                for j, frame in data():
                    frame_fn = f'flow_accel_{j}.npy'
                    if not os.path.isdir(f'{root}{split}/{fn[:-4]}/{frame_fn}'):
                        if backup is not None and \
                                os.path.isdir(f'{backup}{split}/{fn[:-4]}/{frame_fn}'):
                            os.system(f'cp {backup}{split}/{fn[:-4]}/{frame_fn} {root}{split}/{fn[:-4]}/{frame_fn}')
                        else:
                            np.save(f'{root}{split}/{fn[:-4]/{frame_fn}', frame)
                i += 1
                print('%5d: %s' % (i, fn))

def main():
    global root, backup
    root = sys.argv[1] # /home/shared/Moments_in_Time_Mini_Stream/
    if root[-1] != '/':
        root += '/'
    if len(sys.argv) > 2:
        backup = sys.argv[2]
        if backup[-1] != '/':
            backup += '/'
    process_split('training')
    process_split('validation')

if __name__ == '__main__':
    main()
