from PIL import Image
import os
import numpy as np

# In this file, I will implement rotate, fill, transponse, add/minus/identical, and compare similarity
# os.chdir('Project-Code-Python/Problems/Basic Problems B/Basic Problem B-01/')
imgA = np.array(Image.open('A.png'))
imgB = np.array(Image.open('B.png'))
imgC = np.array(Image.open('C.png'))



angels = np.array([0,30,45,60,90])  #remember to fill the background as white.
transpose_cat = [Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM, Image.TRANSPOSE]


#detect line pixel changes from white to black and return the times of changes.
# Just realize, fill is another way to add, no need to do that!
def line_break_detect(line,white):
    return p_start,p_end,breaks,white_or_black

def fill_halo(img):
    line_breaks = np.zeros(img.shape[0],3)
    if np.sum(img[0])/img.size == 255
        white = 1
    for line in img:
        s,e,b,wb = line_break_detect(line,white)
        line_breaks.append([s,e,b,wb])
    patterns = None                         # find a way to identify white-black changes.

# MSE for comparing differences between images.
def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err
