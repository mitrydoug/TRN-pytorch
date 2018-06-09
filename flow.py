import os, sys
from threading import Thread
import time

import cv2
import numpy as np
from queue import Queue

import PIL
from PIL import Image

NUM_THREADS=30
output_queue = Queue()

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
        return self.Q.get()

    def more(self):
        # return True if there are still frames in the queue
        while self.Q.qsize() == 0:
            if self.stopped:
                if self.Q.qsize() == 0:
                    return False
            time.sleep(0.001)
        return True

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True
        self.stream.release()

def compute_flow_for_video(root, split, fn):
    def down(img):
        return np.uint8((img[::2,::2] + img[1::2,1::2]) / 2.)
    video_path = f'{root}{split}/{fn}'
    flows = []
    fvs = FileVideoStream(video_path).start()
    if fvs.more():
        frame1 = down(fvs.read())
    else:
        print(f'Error reading: {video_path}')
        return
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    while fvs.more():
        frame2 = down(fvs.read())
        next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
                prvs, next, None, 0.5, 3, 8, 2, 5, 1.1, 0)
        flow[np.isnan(flow)] = 0.
        flows.append(flow)
        prvs = next
    flows.append(flows[-1])
    
    accels = [np.zeros_like(flows[0])]
    for i, flow in enumerate(flows[1:]):
        prev = flows[i]
        rows = np.maximum(np.minimum(
            np.arange(128).reshape((128, 1)) + prev[:,:,0], 127), 0).astype(np.uint8)
        cols = np.maximum(np.minimum(
            np.arange(128).reshape((1, 128)) + prev[:,:,1], 127), 0).astype(np.uint8)
        flow = flow[rows, cols]
        accel = flow-prev
        accels.append(accel)

    zs = np.zeros((128,128,1))
    for i, (flow, acel) in enumerate(zip(flows, accels)):
        flow = np.minimum(np.maximum((flow + 15.) * 8, 0), 255)
        acel = np.minimum(np.maximum((acel + 30.) * 4, 0), 255)
        x_flow = np.concatenate((flow[:,:,[0]], acel[:,:,[0]], zs), axis=2).astype(np.uint8)
        y_flow = np.concatenate((flow[:,:,[1]], acel[:,:,[1]], zs), axis=2).astype(np.uint8)
        Image.fromarray(x_flow).resize((256, 256), resample=PIL.Image.BILINEAR).save(
                f'{root}{split}/{fn[:-4]}/fx_{i+1:05}.jpg')
        Image.fromarray(y_flow).resize((256, 256), resample=PIL.Image.BILINEAR).save(
                f'{root}{split}/{fn[:-4]}/fy_{i+1:05}.jpg')

root = None
backup = None

def process_list(root, split, lines, labels):
    for line in lines:
        fn, label, _, _ = line.split(',')
        assert (fn[-4:] == '.mp4')
        #num_frames = len([x for x in os.listdir(f'{root}{split}/{fn[:-4]}') if 'flow' in x])
        #if num_frames > 70:
        #    continue
        #x = np.load(f'{root}{split}/{fn[:-4]}/flow.npz')['arr_0']
        #compute_flow_for_video(root, split, fn)
        #if frames is None:
        #    continue
        #for i in range(frames.shape[3]):
        #    np.savez_compressed(f'{root}{split}/{fn[:-4]}/flow_{i+1:05}.npz', frames[:,:,:,i])
        #num_flow = len([x for x in os.listdir(f'{root}{split}/{fn[:-4]}') if 'flow' in x])
        #np.savez_compressed(f'{root}{split}/{fn[:-4]}/flow.npz', precision(frames))
        #num_frames = len([x for x in os.listdir(f'{root}{split}/{fn[:-4]}') if 'flow' not in x])
        #output_queue.put(f'{split}/{fn[:-4]} {num_frames} {labels[label]}\n')
    return

def process_split(split):
    split_file = f'{root}{split}Set.csv'
    frames_file = f'{root}{split}_frames.txt'

    with open(f'{root}moments_categories.txt', 'r') as f:
        labels = dict(tuple(line.strip().split(',')) for line in f)

    i = 0
    
    with open(split_file, 'r') as f:
        lines = f.readlines()

    threads = [None] * NUM_THREADS

    i, M = 0, 100
    while i < len(lines):
        for j, thread in enumerate(threads):
            if thread is None or not thread.is_alive():
                threads[j] = Thread(target=process_list, args=(root, split, lines[i:i+M], labels))
                threads[j].start()
                print(f'Scheduled lines [{i}, {i+M})')
                i += M
                break
        time.sleep(0.1)

    for thread in threads:
        if thread is not None:
            thread.join()

    #with open(frames_file, 'w') as g:
    #    while output_queue.qsize() > 0:
    #        line = output_queue.get()
    #        g.write(line)

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
