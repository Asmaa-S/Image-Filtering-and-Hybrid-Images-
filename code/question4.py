import time
import numpy as np
from scipy import signal
from mpl_toolkits.mplot3d import Axes3D
from helpers import load_image, save_image, my_imfilter
import cv2
import matplotlib.pyplot as plt
from skimage import data, color
import PIL
from PIL import Image
from skimage.transform import rescale, resize, downscale_local_mean
timeMat= np.zeros((7,5))
image = load_image('../data/RISDance.jpg')
imsizes= [(380,684),(1341,745),(1897,1054),(3000,1666),(3800,2100)]
i=0
y=[0,0.25, 1, 2, 5, 8] #in mps
x=[0,3,5,7,9,11,13,15] #ksize
for ksize in range(3, 16, 2):
    for imsize in range (1,6):

        kernel = np.dot(cv2.getGaussianKernel(ksize, 5), cv2.getGaussianKernel(ksize, 5).T)
        image_rescaled =cv2.resize(image, imsizes[imsize-1], interpolation = cv2.INTER_AREA)
        t = time.time()
        filtered = signal.correlate2d(np.mean(image_rescaled,axis=2), kernel)
        timeMat[i, imsize-1] = time.time() - t
    i=i+1
fig = plt.figure(figsize=(6, 3.2))
print(timeMat)
ax = fig.add_subplot(111)
ax.set_title('colorMap')
plt.imshow(timeMat , extent=[3,15,8,0.25])
ax.set_aspect('equal')

cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
cax.get_xaxis().set_visible(False)
cax.get_yaxis().set_visible(False)
cax.patch.set_alpha(0)
cax.set_frame_on(False)
plt.colorbar(orientation='vertical')
plt.show()

plt.show()
