import cv2
import numpy as np
import scipy.ndimage as nd

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
        mask = nd.gaussian_filter(mask, sigma=(1, 1), order=0) # because red isn't necessarily pen mark (e.g. blood cells) --> filter this out
    percentage = np.mean(mask) 

    return percentage


def blackish_percent(img, threshold=(100, 100, 100)):
    r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
    t = threshold
    mask = (r < t[0]) & (g < t[1]) & (b < t[2])
    mask = nd.gaussian_filter(mask, sigma=(1, 1), order=0)
    percentage = np.mean(mask) 

    return percentage


def filter_pen_marked(tile_path, dest):
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
        with open(dest, "a") as text_file:
            text_file.write(tile_path) 


# source https://github.com/gerstung-lab/PC-CHiP/blob/master/inception/preprocess/imgconvert.py
def getGradientMagnitude(im):
    """Get magnitude of gradient for given image.
    Use for filtering blurry tiles"""
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    ddepth = cv2.CV_32F
    dx = cv2.Sobel(im, ddepth, 1, 0)
    dy = cv2.Sobel(im, ddepth, 0, 1)
    dxabs = cv2.convertScaleAbs(dx)
    dyabs = cv2.convertScaleAbs(dy)
    mag = cv2.addWeighted(dxabs, 0.5, dyabs, 0.5, 0)
    return mag
