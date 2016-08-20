"""
    Usage: python size_reduction.py <<path to image file - String>> <<Number of colors for K means cluster - Integer>>
    Example: python size_reduction.py /tmp/hello.png 32
"""

print(__doc__)

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.utils import shuffle
from time import time

from scipy.misc import imread
from scipy.misc import imsave

import sys
import os

command_arguments = sys.argv

picture_file_path = "picture.jpg";

# Number of colors for K means cluster 
n_colors = 32

# Load photo
if len(command_arguments) > 1 and command_arguments[1] is not None:
        picture_file_path = command_arguments[1] 

if len(command_arguments) > 2 and command_arguments[2] is not None:
        n_colors = int(command_arguments[2]) 

print "Reading Image from path ", picture_file_path, " and number of colors is ", n_colors, "\n\n"
picture = imread(picture_file_path)

# Convert to floats instead of the default 8 bits integer coding. Dividing by
# 255 is important so that plt.imshow behaves works well on float data (need to
# be in the range [0-1]
picture = np.array(picture, dtype=np.float64) / 255

# Load Image and transform to a 2D numpy array.
w, h, d = original_shape = tuple(picture.shape)
assert d == 3
image_array = np.reshape(picture, (w * h, d))

print("Fitting model on a small sub-sample of the data")
t0 = time()
image_array_sample = shuffle(image_array, random_state=0)[:1000]
kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)
print("done in %0.3fs." % (time() - t0))

# Get labels for all points
print("Predicting color indices on the full image (k-means)")
t0 = time()
labels = kmeans.predict(image_array)
print("done in %0.3fs." % (time() - t0))


codebook_random = shuffle(image_array, random_state=0)[:n_colors + 1]
print("Predicting color indices on the full image (random)")
t0 = time()
labels_random = pairwise_distances_argmin(codebook_random,
                                          image_array,
                                          axis=0)
print("done in %0.3fs." % (time() - t0))

def recreate_image(codebook, labels, w, h):
    """Recreate the (compressed) image from the code book & labels"""
    d = codebook.shape[1]
    image = np.zeros((w, h, d))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = codebook[labels[label_idx]]
            label_idx += 1
    return image


image_file_name_tokens = os.path.splitext(picture_file_path)
full_path_without_extn = image_file_name_tokens[0]
extn = image_file_name_tokens[1]

imsave(full_path_without_extn +"_1"+ extn, picture)
imsave(full_path_without_extn +"_km"+ extn, recreate_image(kmeans.cluster_centers_, labels, w, h))
imsave(full_path_without_extn +"_random"+ extn, recreate_image(codebook_random, labels_random, w, h))
