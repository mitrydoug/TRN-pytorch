import os, sys
from threading import Thread
import time

import cv2
import numpy as np
from queue import Queue


class FileVideoStream:
    def __init__(self, path, queueSize=90):
        # initialize the file video stream along with the boolean
        # used to indicate if the thread should be stopped or not
        self.stream = cv2.VideoCapture(path)
        self.stopped = False
 
        # initialize the queue used to store frames read from
        # the video file
        self.Q = Queue(maxsize=queueSize)

    def start(self):
        # start a thread to read frames from the file video stream
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        # keep looping infinitely
        while True:
            # if the thread indicator variable is set, stop the
            # thread
            if self.stopped:
                return
 
            # otherwise, ensure the queue has room in it
            if not self.Q.full():
                # read the next frame from the file
                (grabbed, frame) = self.stream.read()
 
                # if the `grabbed` boolean is `False`, then we have
                # reached the end of the video file
                if not grabbed:
                    self.stop()
                    return
 
                # add the frame to the queue
                self.Q.put(frame)

    def read(self):
        # return next frame in the queue
        while self.Q.qsize() == 0:
            time.sleep(0.001)
        return self.Q.get()

    def more(self):
        # return True if there are still frames in the queue
        return not self.stopped

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True
        self.stream.release()

def compute_flow_for_video(video_fn):
    def avg(fm):
        return (fm[::2,::2] + fm[1::2,1::2]) / 2.
    flows = []
    fvs = FileVideoStream(video_fn).start()
    frame1 = fvs.read()
    prvs = avg(cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY))
    while fvs.more():
        frame2 = fvs.read()
        next = avg(cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY))
        flow = cv2.calcOpticalFlowFarneback(
                prvs, next, None, 0.5, 3, 8, 2, 5, 1.2,
                cv2.OPTFLOW_USE_INITIAL_FLOW)
        flows.append(flow)
        prvs = next
    fvs.stop()
    cv2.destroyAllWindows()
    # assume motion continues
    flows.append(flows[-1])
    accels = []
    for i, flow in enumerate(flows[:-1]):
        next = flows[i+1]
        accels.append(next-flow)
    accels.append(accels[-1])
    result = [np.concatenate([f, a], axis=2) for f, a in zip(flows, accels)]
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
                for j, frame in enumerate(frames):
                    frame_fn = f'flow_{j+1:05}.npy'
                    if not os.path.isdir(f'{root}{split}/{fn[:-4]}/{frame_fn}'):
                        if backup is not None and \
                                os.path.isdir(f'{backup}{split}/{fn[:-4]}/{frame_fn}'):
                            os.system(f'cp {backup}{split}/{fn[:-4]}/{frame_fn} {root}{split}/{fn[:-4]}/{frame_fn}')
                        else:
                            np.save(f'{root}{split}/{fn[:-4]}/{frame_fn}', frame)
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
