import multiprocessing as mp
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def worker_white_tiles(tile_path, q):
    im = cv2.imread(tile_path)
    total_amt_pixels = im.shape[0]*im.shape[1]
    sought = [220,220,220]
    amt_white  = np.count_nonzero(np.all(im>sought,axis=2))
    
    if amt_white > total_amt_pixels/2:
        q.put(tile_path)
    return tile_path

#source https://github.com/gerstung-lab/PC-CHiP/blob/master/inception/preprocess/imgconvert.py
def getGradientMagnitude(im):
    "Get magnitude of gradient for given image"
    im=cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    ddepth = cv2.CV_32F
    dx = cv2.Sobel(im, ddepth, 1, 0)
    dy = cv2.Sobel(im, ddepth, 0, 1)
    dxabs = cv2.convertScaleAbs(dx)
    dyabs = cv2.convertScaleAbs(dy)
    mag = cv2.addWeighted(dxabs, 0.5, dyabs, 0.5, 0)
    return mag

def worker_blurred(tile_path, q, amt=15, perc=0.5, size_px=512):
    grad = getGradientMagnitude(plt.imread(tile_path))
    unique, counts = np.unique(grad, return_counts=True)
    if (counts[np.argwhere(unique<=amt)].sum() < size_px*size_px*perc) == False:
        q.put(tile_path)
    return tile_path

def listener(q, fn):
    '''listens for messages on the q, writes to file. '''

    with open(fn, 'w') as f:
        while 1:
            m = q.get()
            if m == 'kill':
                f.write('killed')
                break
            f.write(str(m) + '\n')
            f.flush()

if __name__ == '__main__':
    to_filter = 'white' # 'white' for white tile filtering or 'blurred' for blurred tiles filtering

    fn = to_filter+'_tile_paths.txt' # file name of file that will be written. Will contain tile paths of white/blurred tiles
    df_patients_labels = pd.read_csv('df_patients_labels.csv')

    # Manager queue
    manager = mp.Manager()
    q = manager.Queue()    
    pool = mp.Pool(mp.cpu_count() + 2)

    # put listener to work first
    watcher = pool.apply_async(listener, (q, fn))

    # fire off workers
    tile_paths = df_patients_labels['tile_path'].values
    jobs = []
    for tile_path in tile_paths:
        if to_filter == 'white':
            job = pool.apply_async(worker_white_tiles, (tile_path, q))
        else:
            job = pool.apply_async(worker_blurred, (tile_path, q))
        jobs.append(job)

    # collect results from the workers through the pool result queue
    for job in jobs: 
        job.get()

    # now we are done, kill the listener
    q.put('kill')
    pool.close()
    pool.join()


