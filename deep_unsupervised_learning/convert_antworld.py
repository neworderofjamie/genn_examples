import cv2
import numpy as np
import matplotlib.pyplot as plt

from glob import glob

uv_green = False
image_files = list(glob("world5000_grid/*.png"))

image_data = [cv2.imread(f) for f in image_files]

data = []
for image in image_data:
    green = image[:,:,1]

    uv = (image[:,:,0] == 191) & (image[:,:,1] == 191) & (image[:,:,2] == 191)
    if uv_green:
        data.append(np.dstack((green, uv.astype(np.uint8) / 4)))
    else:
        green[uv] = 0
        data.append(green)

data = np.stack(data)
print(data.shape)

if uv_green:
    data.tofile("uv_green_grid.bin")
else:
    data.tofile("green_grid.bin")
