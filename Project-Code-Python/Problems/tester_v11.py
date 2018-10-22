from PIL import Image
from PIL import ImageFilter
import numpy as np
import os, glob
import time

# home = 'C:/Users/s2235/PycharmProjects/KBAI_project/Project-Code-Python/'
home = 'd:/PycharmProjects/KBAI_project/Project-Code-Python/'
problemset = 'Basic Problems C/Basic Problem C-07/'
os.chdir(home + 'Problems/' + problemset)

images = {}
img = {}
imgX = {}

for filename in glob.glob('*.png'):
    if len(filename) == 5:
        images[filename[0]] = home + 'Problems/' + problemset + filename

for key, value in images.items():

    if key.isalpha() == True:
        image_array = Image.open(images[key])
        # image_array = np.array(Image.open(images[key]))
        img[key] = image_array
        #img[key] = img_threshold(image_array, 128)
    # else key.isdigit() ==True:
    else:
        imgX[key] = Image.open(images[key])
        # imgX[key] = np.array(Image.open(images[key]))
        #imgX[key] = img_threshold(np.array(Image.open(images[key])), 128)

def mse(imageA, imageB):        # eat up 2 arrays.
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    imgarrayA = np.array(imageA.filter(ImageFilter.BLUR))
    imgarrayB = np.array(imageB.filter(ImageFilter.BLUR))
    err = np.sum((imgarrayA.astype("float") - imgarrayB.astype("float")) ** 2)
    err /= float(imgarrayA.shape[0] * imgarrayA.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err

def self_symmetric (img,threshold):   # eat up a image object.
    ans = 0
    symmetry_set = [Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM]
    for trans in symmetry_set:
        if mse(img,img.transpose(trans))<threshold:
            ans = 1
            break
    return ans

def combine_figures(img_dic,imgX):      # update for 3x3 problem
    w,h = img_dic['A'].size
    combined_img = Image.new('RGBA', (w*3,h*3))
    combined_img.paste(img_dic['A'],(0,0))
    combined_img.paste(img_dic['B'], (w, 0))
    combined_img.paste(img_dic['C'], (2*w, 0))
    combined_img.paste(img_dic['D'], (0, h))
    combined_img.paste(img_dic['E'], ( w, h))
    combined_img.paste(img_dic['F'], (2 * w, h))
    combined_img.paste(img_dic['G'], (0, 2*h))
    combined_img.paste(img_dic['H'], ( w, 2* h))
    combined_img.paste(imgX, (2*w, 2*h))
    return combined_img

c = combine_figures(img,imgX['2'])

ans = self_symmetric(c,1000)
print(ans)
