import multiprocessing as mp
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.ndimage as nd

# identifying 'blurriness' of tile
# source https://github.com/gerstung-lab/PC-CHiP/blob/master/inception/preprocess/imgconvert.py
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


# parallel worker that writes tile paths of white tiles
def worker_white_tiles(tile_path, q):
    im = cv2.imread(tile_path)
    total_amt_pixels = im.shape[0]*im.shape[1]
    sought = [220,220,220]
    amt_white  = np.count_nonzero(np.all(im>sought,axis=2))
    
    if amt_white > total_amt_pixels/2:
        q.put(tile_path)
    return tile_path


# parallel worker for blurred tiles
def worker_blurred(tile_path, q, amt=15, perc=0.5, size_px=512):
    grad = getGradientMagnitude(plt.imread(tile_path))
    unique, counts = np.unique(grad, return_counts=True)
    if (counts[np.argwhere(unique<=amt)].sum() < size_px*size_px*perc) == False:
        q.put(tile_path)
    return tile_path


# identify percentage of black in tile
def blackish_percent(img, threshold=(100, 100, 100)):
    r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
    t = threshold
    mask = (r < t[0]) & (g < t[1]) & (b < t[2])
    mask = nd.gaussian_filter(mask, sigma=(1, 1), order=0)
    
    percentage = np.mean(mask)
    return percentage


# identify shades of red, green, blue to filter (for pen marks)
# ref: https://github.com/CODAIT/deep-histopath/blob/master/deephistopath/wsi/filter.py
PENS_RGB = {
    "red": [
        (150, 80, 90),
        (110, 20, 30),
        (185, 65, 105),
        (195, 85, 125),
        (220, 115, 145),
        (125, 40, 70),
        (200, 120, 150),
        (100, 50, 65),
        (85, 25, 45),
    ],
    "green": [
        (150, 160, 140),
        (70, 110, 110),
        (45, 115, 100),
        (30, 75, 60),
        (195, 220, 210),
        (225, 230, 225),
        (170, 210, 200),
        (20, 30, 20),
        (50, 60, 40),
        (30, 50, 35),
        (65, 70, 60),
        (100, 110, 105),
        (165, 180, 180),
        (140, 140, 150),
        (185, 195, 195),
    ],
    "blue": [
        (60, 120, 190),
        (120, 170, 200),
        (120, 170, 200),
        (175, 210, 230),
        (145, 210, 210),
        (37, 95, 160),
        (30, 65, 130),
        (130, 155, 180),
        (40, 35, 85),
        (30, 20, 65),
        (90, 90, 140),
        (60, 60, 120),
        (110, 110, 175),
    ],
}


def pen_percent(img, pen_color):
    r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
    thresholds = PENS_RGB[pen_color]

    if pen_color == "red":
        t = thresholds[0]
        mask = (r > t[0]) & (g < t[1]) & (b < t[2])

        for t in thresholds[1:]:
            mask = mask | ((r > t[0]) & (g < t[1]) & (b < t[2]))

    elif pen_color == "green":
        t = thresholds[0]
        mask = (r < t[0]) & (g > t[1]) & (b > t[2])

        for t in thresholds[1:]:
            mask = mask | (r < t[0]) & (g > t[1]) & (b > t[2])

    elif pen_color == "blue":
        t = thresholds[0]
        mask = (r < t[0]) & (g < t[1]) & (b > t[2])

        for t in thresholds[1:]:
            mask = mask | (r < t[0]) & (g < t[1]) & (b > t[2])

    else:
        raise Exception(f"Error: pen_color='{pen_color}' not supported")

    if pen_color == "red": 
        mask = nd.gaussian_filter(mask, sigma=(1, 1), order=0) # filter out e.g. detected loose blood cells (which we don't want to consider as "pen mark")
    percentage = np.mean(mask) 

    return percentage

# parallel worker for penmark tiles
def worker_penmarked(tile_path, q):
    write = 0
    img = cv2.imread(tile_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for c in ['green', 'red']: 
        if (pen_percent(img, c) >= 0.05):
            write = 1
    if (blackish_percent(img) >= 0.05):
        write = 1
    if (pen_percent(img, 'blue') >= 0.25):
        write = 1
    if write == 1:
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
    to_filter = 'white' # 'white' for white tile filtering, 'blurred' for blurred tiles filtering or 'penmarks' for pen marks

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
        elif to_filter == 'blurred':
            job = pool.apply_async(worker_blurred, (tile_path, q))
        elif to_filter == 'penmarks':
            job = pool.apply_async(worker_penmarked, (tile_path, q))
        else:
            print('to_filter wrong value, should be "white", "blurred" or "penmarks"')
            exit()

        jobs.append(job)

    # collect results from the workers through the pool result queue
    for job in jobs: 
        job.get()

    # now we are done, kill the listener
    q.put('kill')
    pool.close()
    pool.join()


