# Image processing
# Run SVD on array
# grayscale photo download from : www.cs.princeton.edu/courses/archive/fall15/cos521/image.jpg

# Question 1: What is the value of k such that a rank k approximation gives a reasonable approximation (visually) to the image?
# Question 2: What value of k gives an approximation that looks high quality to your eyes?

from scipy import misc
import numpy as np

myArray = misc.imread('image.jpg')

print(myArray.shape)

U, s, V = np.linalg.svd(myArray, full_matrices=True)
print(U.shape, s.shape, V.shape)
