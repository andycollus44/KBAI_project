from PIL import Image
from PIL import ImageFilter
import numpy as np
import os, glob
import time
#
# home = 'C:/Users/s2235/PycharmProjects/KBAI_project/Project-Code-Python/'
home = 'd:/PycharmProjects/KBAI_project/Project-Code-Python/'
problemset = 'Basic Problems C/Basic Problem C-12/'
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

def dark_ratio(img,thres):
    imgarray = np.array(img)
    img_tmp = imgarray.copy()
    img_tmp[imgarray > thres] = 0  # white to 0
    img_tmp[imgarray <= thres] = 1  # black to 1
    return np.sum(img_tmp)

def dark_center(img,thres):
    # add the rows and average them.
    imgarray = np.array(img)
    img_tmp = imgarray.copy()
    img_tmp[imgarray > thres] = 0  # white to 0
    img_tmp[imgarray <= thres] = 1  # black to 1
    rows = np.sum(np.sum((img_tmp), axis=0),axis = 1)
    cols = np.sum(np.sum((img_tmp), axis=1),axis = 1)
    return (weighted_center(rows),weighted_center(cols))

def weighted_center(row_array):      # pass a list
    sum = 0
    row_list = list(row_array)
    for i in range(len(row_list)):
        sum = sum+ row_list[i]*(i+1)
    return sum/np.sum(row_list)

def figure_sum(img_dict):
    w, h = img_dict['A'].size
    sum_img = np.zeros((w,h,4))
    for key,value in img_dict.items():
        sum_img = sum_img + np.array(value)
    return sum_img

def calc_diff(list,value):          # the diff of dark list
    list = np.array(list)
    value = np.array(value)
    exp1 = 2*list[5]-list[2]
    exp2 = 2*list[7]-list[6]
    exp3 = 2*list[4]-list[0]
    return (value-exp1)**2+(value-exp2)**2+(value-exp3)**2


def check_dark_increase(list, incre_factor):  # check if there's an increment in dark_ratio

    if np.abs(int(list[1]) - int(list[0])) < incre_factor * int(list[0]):
        return False
    else:
        return True
# dark_center(img['B'],128)

dark_list = []
dark_center_list = []
keys = ['A','B','C','D','E','F','G','H']
for key in keys:
    dark_list.append(dark_ratio(img[key], 128))             # put dark ratio values into a list
    dark_center_list.append(dark_center(img[key], 128))


# print(dark_list)
def test_dark_ratio(img, imgX, thres):
    dark_list = []
    dark_center_list = []
    # # keys = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    # keys = img.keys()
    # for k in sorted(img.iterkeys()):
    #     dark_list.append(dark_ratio(img[k], thres))             # put dark ratio values into a list
    #     dark_center_list.append(dark_center(img[k], thres))
    for key, value in sorted(img.items()):
        dark_list.append(dark_ratio(img[key], thres))  # put dark ratio values into a list
        dark_center_list.append(dark_center(img[key], thres))

    # if len(dark_list) == 0:
    #     print('empty dark list')
    # # elif check_dark_increase(dark_list, 0.1) == False:
    # #     return 0
    min_center = 1000
    min_diff = 10000000000
    ans = 0
    for key, value in imgX.items():

        diff = calc_diff(dark_list, dark_ratio(value, thres))
        center_dff = calc_diff(dark_center_list,dark_center(value,thres))
        if type(diff) == type((1, 1)):
            diff = np.sum(diff)
        if diff < min_diff:
            min_diff = diff
            ans = int(key)
    return ans


def test_horiz_switch(img, imgX, thres):
    for key, value in imgX.items():
        if mse(horiz_switch(value), img['G']) < thres:
            return key

def horiz_switch(img):

    imgarray = np.array(img)
    w, h, c = imgarray.shape
    img_tmp = imgarray.copy()
    img_tmp[:,0:int(h/2-1),:] = imgarray[:,int(h/2):-1,:]
    img_tmp[:, int(h / 2):-1 :] = imgarray[:,0:int(h/2)-1,:]

    return Image.fromarray(img_tmp)



# print(dark_list)
print(dark_center(imgX['2'],128))

# combine_figures(img, imgX['6']).show()

# for key, value in imgX.items():
#     if self_symmetric(combine_figures(img, imgX[key]), 1000) == True:  # the threshold should be < 3000
#         answer = key

answer = test_dark_ratio(img, imgX, 128)
if answer == 0:
    answer = test_horiz_switch(img, imgX, 2000)
    print(answer)

# weighted_center(img['A'],128)