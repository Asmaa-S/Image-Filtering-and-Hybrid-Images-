import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from helpers import load_image, save_image, my_imfilter, show_images,fft_filter
from skimage.transform import rescale

try_it_out_with_gray = False
save_outputs = True
show_outputs = True

# creating a directory for the resulting images to go to
resultsDir = '..' + os.sep + 'results'
if not os.path.exists(resultsDir):
    os.mkdir(resultsDir)

# loading an image to try the filters on and resizing it (downsampling)
test_image = load_image('../data/cat.bmp')
test_image = rescale(test_image, 0.7, mode='reflect', multichannel=True)
if try_it_out_with_gray:
    test_image = test_image.mean(axis=2)


laplacian_filter = np.asarray([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
large_1d_blur_filter = cv2.getGaussianKernel(5, 5)
large_blur_filter = np.dot(large_1d_blur_filter, large_1d_blur_filter.T)

# # Im gonna try the fft method filter on the gaussian blur, and the laplacian filter
out = my_imfilter(test_image, large_blur_filter, domain='fft', visualize_fft=True, im_name_fft="gauss_blur")
out2 = my_imfilter(test_image, laplacian_filter, domain='fft', visualize_fft=True, im_name_fft="laplacian")


'''
Identity filter
This filter should do nothing regardless of the padding method you use.
'''
identity_filter = np.asarray([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
identity_image = my_imfilter(test_image, identity_filter, 'zero')



'''
Small blur with a box filter
This filter should remove some high frequencies.
# '''
blur_filter = np.ones((3, 3), dtype=np.float32)
blur_filter /= np.sum(blur_filter, dtype=np.float32)  # making the filter sum to 1
blur_image = my_imfilter(test_image, blur_filter, 'zero')



'''
Large blur
This blur would be slow to do directly, so we instead use the fact that Gaussian blurs are separable and blur sequentially in each direction.
'''

## Slow (naive) version of large blur
import time

t = time.time()
large_blur_image = my_imfilter(test_image, large_blur_filter, 'reflect')
t = time.time() - t
print('{:f} seconds'.format(t))


#

##
#
# '''
# Oriented filter (Sobel operator)
# '''
sobel_filter = np.asarray([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                          dtype=np.float32)  # should respond to horizontal gradients
sobel_image = my_imfilter(test_image, sobel_filter, 'zero')

# # 0.5 added because the output image is centered around zero otherwise and mostly black
sobel_image = np.clip(sobel_image + 0.5, 0.0, 1.0)


#

#
# '''
# High pass filter (discrete Laplacian)
# '''
laplacian_image = my_imfilter(test_image, laplacian_filter, 'zero')
# # added because the output image is centered around zero otherwise and mostly black
laplacian_image = np.clip(laplacian_image + 0.5, 0.0, 1.0)




# # High pass "filter" alternative: subtracting the low frequency component from the original image
high_pass_image = test_image - blur_image
high_pass_image = np.clip(high_pass_image + 0.5, 0.0, 1.0)



if save_outputs:
    done = save_image('../results/identity_image.jpg', identity_image)
    done = save_image(resultsDir + os.sep + 'high_pass_image.jpg', high_pass_image)
    done = save_image(resultsDir + os.sep + 'laplacian_image.jpg', laplacian_image)
    done = save_image(resultsDir + os.sep + 'blur_image.jpg', blur_image)
    done_gaus = save_image(resultsDir + os.sep + 'CVgaus_blur_image.jpg', large_blur_image)
    done = save_image(resultsDir + os.sep + 'sobel_image.jpg', sobel_image)

if show_outputs:
    images = [identity_image, high_pass_image, laplacian_image, blur_image, large_blur_image]
    titles = ['identity_image', 'high_pass_by_subtraction', 'laplacian_image', 'small_blur_with_box', "large_blur_with gauss"]
    show_images(images, titles)
