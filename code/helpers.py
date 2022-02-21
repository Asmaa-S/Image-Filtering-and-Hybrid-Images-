# Project Image Filtering and Hybrid Images Stencil Code
# Based on previous and current work
# by James Hays for CSCI 1430 @ Brown and
# CS 4495/6476 @ Georgia Tech
import scipy.fftpack as fp
import numpy as np
from numpy import pi, exp, sqrt
from skimage import io, img_as_ubyte, img_as_float32
from skimage.transform import rescale
from scipy import signal
import cv2
from copy import deepcopy
import itertools
from enum import Enum
import matplotlib.pyplot as plt


def zero_pad(X, pad):
    """
    Pad with zeros all images of the dataset or batch X.

    Arguments:
      - X -- numpy array of shape (n, h, w, c) representing a batch of n images
      - pad -- integer, padding amount

    Returns:
      - X_pad -- padded images of shape (n, h + 2*pad, w + 2*pad, c)
    """
    # we will pad 3 dimensions
    extraRows = pad
    extraCols = pad

    padding_size = ((extraRows, extraRows), (extraCols, extraCols), (0, 0))

    X_pad = []

    for image in X:
        padded_image = np.pad(image, padding_size, 'constant', constant_values=0)
        X_pad.append(padded_image)

    return np.array(X_pad)

def conv_single_step(input_slice, W, b, stride=(1, 1), reduce_channels=False):
    """
    Apply one filter on a single slice of the layer's input.

    Arguments:
      - input_slice -- slice of input data of shape (f, f, c)
      - W -- Weight parameters of the filter - matrix of shape (f, f, c)
      - b -- Bias parameter associated with the filter - matrix of shape (1, 1, 1)

    Returns:
      - Z -- result of convolving the sliding filter with a slice of the input data
              - a scalar value
    """
    # the padded region's size
    kernRows = W.shape[0]
    kernCols = W.shape[1]

    # original image 2D size
    n_rows = input_slice.shape[0]
    n_cols = input_slice.shape[1]

    # otput size
    h_out = int((n_rows - kernRows) / stride[0] + 1)
    w_out = int((n_cols - kernCols) / stride[1] + 1)

    # initializing a mold for the output image
    Z = np.zeros((h_out, w_out, input_slice.shape[2]))

    indices = itertools.product(np.arange(n_rows), np.arange(n_cols))
    out_indices = list(itertools.product(np.arange(h_out), np.arange(w_out)))

    k = 0
    for i_in_orig, j_in_orig in indices:

        # check bounds
        if i_in_orig + kernRows > n_rows or j_in_orig + kernCols > n_cols:
            continue

        # check strides
        if i_in_orig % stride[0] != 0 or j_in_orig % stride[1] != 0:
            continue

        # looping over the image in blocks centered at each pixel
        img_block = input_slice[i_in_orig: i_in_orig + kernRows, j_in_orig:j_in_orig + kernCols, :]

        # constructing the output pixel by pixel from the result of filtering the corresponding block
        i, j = out_indices[k]
        Z[i, j, :] = np.multiply(img_block, W).sum(axis=1).sum(axis=0) + b
        k += 1

    if reduce_channels:
        Z = Z.sum(axis=2)

    return Z


def conv_forward(X, W, b, stride=(1, 1), pad=0):
    """
    Implements the forward propagation for a convolution function

    Arguments:
    X -- output activations of the previous layer, numpy array of shape (n, h_in, w_in, c_in)
    W -- Weights, numpy array of shape (f, f, c_in, n_filter)
    b -- Biases, numpy array of shape (1, 1, 1, n_filter)
    stride -- integer
    pad -- integer

    Returns:
    Z -- conv output, numpy array of shape (n, h_out, w_out, c_out)
    """

    # Retrieve dimensions from X's shape
    n_inputs, rows, cols, channels = X.shape

    # Retrieve dimensions from W's shape
    filt_rows, filt_cols, filt_channels,n_filters = W.shape

    # Compute the dimensions of the CONV output volume
    # z.shape = n_inputs, h_out, w_out, n_filters where:
    h_out = int((rows - filt_rows+ 2 * pad) / stride[0] + 1)
    w_out = int((cols - filt_cols+ 2 * pad) / stride[1] + 1)
    Z = []

    # Create X_pad by padding X

    if pad > 0:
        x_pad = zero_pad(X, pad)
    else:
        x_pad = X.copy()

    for x in x_pad:
        one_sample_all_filters = []
        for i in range(n_filters):
            kernel_w = W[:, :, :, i]
            kernel_b = b[:, :, :, i]
            out = conv_single_step(x, kernel_w, kernel_b, stride, reduce_channels=True)
            one_sample_all_filters.append(out)

        Z.append(np.reshape(one_sample_all_filters, (3,3,n_filters)))

    Z = np.array(Z)
    # Making sure your output shape is correct
    assert (Z.shape == (n_inputs, h_out, w_out, n_filters))

    return Z



def show_images(images, titles):
    """
    This function is used to show image(s) with titles by sending an array of images and an array of associated titles.
    images[0] will be drawn with the title titles[0] if exists
    :param images: an array of images or a single image
    :param titles: an array of titles or a single title
    """
    # handling different inputs [list or scalars]
    try:
        iter(images)
    except:
        images = [images]
    try:
        iter(titles)
    except:
        titles = [titles]

    assert len(images) == len(titles)
    for title in titles:
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)

    for title, img in zip(titles, images):
        cv2_compatible = cv2.cvtColor(img.astype('float32'), cv2.COLOR_BGR2RGB)
        cv2.imshow(title, cv2_compatible)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def fft_filter(img, kernel, Visualize_process=False, img_name=''):
    """
    This function filters an image with a given filter kernel in the frequency domain.
    :param Visualize_process: set True if you want to see the spectrum of the image, the mask, & the filtered image
    :param img: the image to be filtered
    :param kernel: the filter kernel used in the filtering
    :return: the filtered image
    """

    def filter_one_channel(img, mask):
        # step1: getting the spectrum of the image: dft is a 3D array where the third dimension has two channels; the
        # first is the real part and the 2nd is the imaginary part
        dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
        # normally the spectrum starts at the leftmost upper corner of the array, we would like to shift it to the
        # so that the low frequencies would be centered at the origin and the higher frequencies on the outside
        shifted_dft = np.fft.fftshift(dft)

        # here we're calculating the magnitude spectrum of the image from the real and imaginary parts for visualization
        image_spectrum = 20 * np.log(cv2.magnitude(shifted_dft[:, :, 0], shifted_dft[:, :, 1]))

        # now we will apply the filter mask on both the real and imaginary parts
        filtered_shifted_dft = shifted_dft * np.dstack((mask, mask))

        # np.log spits out a warning whenever it gets a zero because the log of zero is undefined. To avoid this,
        # we're adding a very small number in place of the zeros (whose value won't affect the final result)
        zero_dummy = np.ones_like(filtered_shifted_dft) * 10 ** -12
        filtered_shifted_dft = filtered_shifted_dft + zero_dummy

        # getting the spectrum of the filtered result for visualization
        filtered_spectrum = 20 * np.log(cv2.magnitude(filtered_shifted_dft[:, :, 0], filtered_shifted_dft[:, :, 1]))

        # undoing the dft shift
        inverse_shift = np.fft.ifftshift(filtered_shifted_dft)

        # undoing the dft (inverse dft) - to get the filtered image back
        filtered_image = cv2.idft(inverse_shift, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
        # filtered_image = cv2.magnitude(idft[:, :, 0], idft[:, :, 1])

        return filtered_image, image_spectrum, filtered_spectrum

    gray = len(img.shape) == 2

    r, c = (img.shape[0], img.shape[1])
    kr, kc = kernel.shape
    padded = np.pad(kernel, ((0, r - kr), (0, c - kc)), 'constant', constant_values=0)
    dft_k = cv2.dft(np.float32(padded), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_k_sh = np.fft.fftshift(dft_k)
    mask = cv2.magnitude(dft_k_sh[:, :, 0], dft_k_sh[:, :, 1])

    if gray:
        filtered_image, image_spectrum, filtered_spectrum = filter_one_channel(img, mask)
    else:
        f1, s1, fs1 = filter_one_channel(img[:, :, 0], mask)
        f2, s2, fs2 = filter_one_channel(img[:, :, 1], mask)
        f3, s3, fs3 = filter_one_channel(img[:, :, 2], mask)
        filtered_image = np.dstack((f1, f2, f3))
        image_spectrum = np.dstack((s1, s2, s3))
        filtered_spectrum = np.dstack((fs1, fs2, fs3))
        mask = np.dstack((mask, mask, mask))

    if Visualize_process:
        cv2.normalize(image_spectrum, image_spectrum, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(filtered_spectrum, filtered_spectrum, 0, 1, cv2.NORM_MINMAX)

        stacked = np.hstack((image_spectrum, mask, filtered_spectrum))
        save_image(f"../results/{img_name}_FFT_img_spectrum+filter+output_spectrum.jpg", stacked)
        save_image(f"../results/{img_name}_fft_output.jpg", filtered_image)
        show_images([img, image_spectrum, mask, filtered_spectrum, filtered_image],
                    ['original', 'image spectrum', 'filter mask', 'filtered spectrum', 'filtered image'])

    return filtered_image


def correlate_one_channel(original_image, padded_image, filter_kernel):
    """
    a helper function to "my_imfilter". convolves a 2d image with a given filter kernel
    :param original_image: 2d image to be filtered
    :param padded_image: the same image padded to the appropriate size
    :param filter_kernel: the filter kernel to be used in the correlation/convolution process
    :return: the filtered output image resulting from the convolution/correlation
    """

    # the padded region's size
    extraRows = filter_kernel.shape[0] // 2
    extraCols = filter_kernel.shape[1] // 2

    # original image 2D size
    n_rows = original_image.shape[0]
    n_cols = original_image.shape[1]

    # initializing a mold for the output image
    filtered_image = np.zeros((n_rows, n_cols))

    indices = itertools.product(np.arange(n_rows), np.arange(n_cols))
    for i_in_orig, j_in_orig in indices:
        anchor_i = i_in_orig + extraRows
        anchor_j = j_in_orig + extraCols
        # looping over the image in blocks centered at each pixel
        img_block = padded_image[anchor_i - extraRows: anchor_i + extraRows + 1,
                    anchor_j - extraCols:anchor_j + extraCols + 1]
        # constructing the output pixel by pixel from the result of filtering the corresponding block
        filtered_image[i_in_orig][j_in_orig] = np.multiply(img_block, filter_kernel).sum()

    return filtered_image


def my_imfilter(image: np.ndarray, filter_kernel: np.ndarray, padding_method='zero', method='convolution',
                domain='spatial', im_name_fft="", visualize_fft=False):
    """
    filters the given image with the filter kernel. Supports grayscale and rgb.
    :param visualize_fft: a visualization verbose to show the spectra of images during fft calculations
    :param im_name_fft: file name of the image to save visualized results
    :param domain: 'spatial': (default) filter using regular convolution/correlation in the time domain
                    'fft': filter in the frequency domain using fft
    :param image: input image, gray scale or 3 channel RGB/BGR
    :param filter_kernel: filter kernel
    :param padding_method: supports 'zero', 'reflect', 'edge' or 'mean' padding.
    :param method: filter application method, 'convolution' or 'correlation'
    :return: the filtered image
    """
    # checking the "domain" input argument
    assert domain in ['spatial',
                      'fft'], 'choose either "spatial", or "fft" for the "domain=" argument'

    # checking the "method" input argument
    assert method in ['convolution',
                      'correlation'], 'choose either "convolution", or "correlation" for the "method=" argument'

    # checking the "padding_method" argument
    assert padding_method in ['zero', "edge", "mean", 'reflect'], 'choose either "zero", "reflect", "edge" or "mean" ' \
                                                                  'for the "padding_method=" argument '

    # Checking the filter_kernel size:
    size = filter_kernel.shape
    if size[0] % 2 == 0 or size[1] % 2 == 0:
        print('Invalid Filter Size! filter_kernel dimensions must be odd')
        return

    # check if image is grayscale
    gray = len(image.shape) == 2

    if method == 'convolution':
        # convolution uses the same technique as correlation but Flips the filter_kernel 180 deg
        filter_kernel = np.flipud(np.fliplr(filter_kernel))
        # note that if method equal 'correlation' we won't be flipping the kernel

    if domain == 'spatial':

        # calculating how many extra rows & cols of padding are needed
        ksize = filter_kernel.shape
        extraRows = ksize[0] // 2
        extraCols = ksize[1] // 2

        # if image is gray scale we will pad two dimensions only, otherwise we will pad 3 dimensions
        padding_size = ((extraRows, extraRows), (extraCols, extraCols)) if gray else (
            (extraRows, extraRows), (extraCols, extraCols), (0, 0))

        if padding_method == 'zero':
            # Pad the image with Zeros
            padded_image = np.pad(image, padding_size, 'constant', constant_values=0)

        else:
            padded_image = np.pad(image, padding_size, padding_method)

        # Applying the filter_kernel
        if gray:
            filtered_image = correlate_one_channel(image, padded_image, filter_kernel)

        else:
            ch1 = correlate_one_channel(image, padded_image[:, :, 0], filter_kernel)
            ch2 = correlate_one_channel(image, padded_image[:, :, 1], filter_kernel)
            ch3 = correlate_one_channel(image, padded_image[:, :, 2], filter_kernel)
            filtered_image = np.stack((ch1, ch2, ch3), axis=2)

    else:
        filtered_image = fft_filter(image, filter_kernel, Visualize_process=visualize_fft, img_name=im_name_fft)

    return filtered_image


def gen_hybrid_image(image1: np.ndarray, image2: np.ndarray, cutoff_frequency: float, ksize, Domain='spatial'):
    """
   Inputs:
   - image1 -> The image from which to take the low frequencies.
   - image2 -> The image from which to take the high frequencies.
   - cutoff_frequency -> The standard deviation, in pixels, of the Gaussian
                         blur that will remove high frequencies.
   -method-> convolution method, 'spatial' for regular spatial domain convolution or 'fft' for frequency domain.
   """
    # checking on the image shape compatibility
    assert image1.shape == image2.shape, "images must have the same shape"

    # check method input
    assert Domain in ['spatial', 'fft'], 'set Domain as either "spatial", or "fft"'

    # generate a 2D Gaussian kernel- to serve as LPF (blur)
    kernel = np.dot(cv2.getGaussianKernel(ksize, cutoff_frequency), cv2.getGaussianKernel(ksize, cutoff_frequency).T)

    # extracting the low and high frequencies of the two images
    low_frequencies = my_imfilter(image1, kernel, domain=Domain)  # dog
    high_frequencies = image2 - my_imfilter(image2, kernel, domain=Domain)  # cat

    # (3) Combine the high frequencies and low frequencies
    # Your code here #
    hybrid_image = low_frequencies + high_frequencies

    hybrid_image = np.clip(hybrid_image, 0, 1)

    # # 0.5 added to the HF image because the output is centered around zero otherwise and mostly black
    # # This is just for the saved HF image to be pretty
    high_frequencies = np.clip(high_frequencies + 0.5, 0.0, 1.0)
    return low_frequencies, high_frequencies, hybrid_image


def vis_hybrid_image(hybrid_image: np.ndarray):
    """
  Visualize a hybrid image by progressively downsampling the image and
  concatenating all of the images together.
  """
    scales = 5
    scale_factor = 0.5
    padding = 5
    original_height = hybrid_image.shape[0]
    num_colors = 1 if hybrid_image.ndim == 2 else 3

    output = np.copy(hybrid_image)
    cur_image = np.copy(hybrid_image)
    for scale in range(2, scales + 1):
        # add padding
        output = np.hstack((output, np.ones((original_height, padding, num_colors),
                                            dtype=np.float32)))
        # downsample image
        cur_image = rescale(cur_image, scale_factor, mode='reflect', multichannel=True)
        # pad the top to append to the output
        pad = np.ones((original_height - cur_image.shape[0], cur_image.shape[1],
                       num_colors), dtype=np.float32)
        tmp = np.vstack((pad, cur_image))
        output = np.hstack((output, tmp))
    return output


def load_image(path):
    return img_as_float32(io.imread(path))


def save_image(path, im):
    try:
        return io.imsave(path, img_as_ubyte(np.round(im.copy(), 6)))
    except:
        im[np.where(im>1)] =1
        im[np.where(im<-1)] =-1
        return io.imsave(path, img_as_ubyte(np.round(im.copy(), 6)))

