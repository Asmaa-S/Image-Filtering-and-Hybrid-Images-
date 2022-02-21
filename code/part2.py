import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from helpers import vis_hybrid_image, load_image, save_image, show_images, gen_hybrid_image, fft_filter

import cv2


# trying out my images
cameron = load_image('../data/cameron.jpg')
zabu = load_image('../data/zabu.jpg')
cutoff_frequency = 10
size = 23

low_frequencies, high_frequencies, cambu = gen_hybrid_image( zabu,cameron, cutoff_frequency, size, Domain='fft')
vis = vis_hybrid_image(cambu)
my_photos = np.hstack((cameron, zabu, cambu))
save_image('../results/cameron+zabu=cambu.jpg', my_photos)
save_image('../results/my_images_hybridized.jpg', cambu)
save_image('../results/Cambu_scales.jpg', vis)



show_plots = False
try_it_out_with_gray = False
# Read images and convert to floating point format
image1 = load_image('../data/dog.bmp')
image2 = load_image('../data/cat.bmp')

if try_it_out_with_gray:
    image1 = image1.mean(axis=2)
    image2 =image2.mean(axis=2)

# Dog/cat
cutoff_frequency = 6
size = 23

# Merging pictures
low_frequencies, high_frequencies, hybrid_image = gen_hybrid_image(image1, image2, cutoff_frequency, size, Domain='fft')



vis = vis_hybrid_image(hybrid_image)

save_image('../results/low_frequencies.jpg', low_frequencies)
save_image('../results/high_frequencies.jpg', high_frequencies)
save_image('../results/hybrid_image.jpg', hybrid_image)
save_image('../results/hybrid_image_scales.jpg', vis)


if show_plots:
    show_images([low_frequencies, high_frequencies, hybrid_image, vis],
                ["low_frequencies", "high_frequencies", "hybrid_image", "visualized"])

