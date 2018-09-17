# tester for find the transformation which gets min(B-A), and then perform on C to find a most likely X.

from PIL import Image
import numpy as np
import os, glob
import time

#home = 'C:/Users/s2235/PycharmProjects/KBAI_project/Project-Code-Python/'
home = 'd:/PycharmProjects/KBAI_project/Project-Code-Python/'
problemset = 'Basic Problems B/Basic Problem B-04/'
os.chdir(home + 'Problems/' + problemset)

images = {}
img = {}
imgX = {}

def img_threshold(img, thres):      # better for subtraction
    img_tmp = np.array(img)
    img_tmp[img > thres] = 2  # white to 1
    img_tmp[img <= thres] = 1  # black to 2
    return img_tmp

for filename in glob.glob('*.png'):
    if len(filename) == 5:
        images[filename[0]] = home + 'Problems/' + problemset + filename

for key, value in images.items():

    if key.isalpha() == True:
        image_array = np.array(Image.open(images[key]))
        img[key] = image_array
        #img[key] = img_threshold(image_array, 128)
    # else key.isdigit() ==True:
    else:
        imgX[key] = np.array(Image.open(images[key]))
        #imgX[key] = img_threshold(np.array(Image.open(images[key])), 128)

angel_set = np.array([0])
transpose_set = [-1, Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM, Image.TRANSPOSE, Image.ROTATE_90, Image.ROTATE_180,
                 Image.ROTATE_270]

def get_mse_transform(imgA, imgB, angel_set, transpose_set):     #transform 2 arrays and find mse follow the transform
    trans_array = np.zeros((len(angel_set),len(transpose_set)))  # store the  tmpTABs
    for i,angel in enumerate(angel_set):
        for j,trans in enumerate(transpose_set):
            im = Image.fromarray(imgA)  # im is an image, img is an array
            try:  # allow no transpose
                im = im.transpose(trans).rotate(angel)
            except ValueError:
                im = im.rotate(angel)

            tmpTAB = mse(np.array(im), np.array(imgB))  # T(A)-B
            trans_array[i][j] = tmpTAB

    return trans_array

def get_least_index(trans_array,threshold):
    ind = np.where(trans_array <=threshold)     #return all the values!
    return (ind[0][0],ind[1][0])

def perform_transform(img,ind_tuple,angel_set,transpose_set):       #return an Image
    im = Image.fromarray(img)  # im is an image, img is an array
    try:  # allow no transpose
        t_img = im.transpose(transpose_set[ind_tuple[1]]).rotate(angel_set[ind_tuple[0]])
    except ValueError:
        t_img = im.rotate(angel_set[ind_tuple[0]])
    return t_img

# # test
# t_a = get_mse_transform(img['C'], imgX['2'], angel_set, transpose_set)
# ind = get_least_index(t_a)
# perform_transform(img['C'],ind,angel_set,transpose_set).show()


def test_transform_X(imgX, T_img):  # imgX dic of X.png,
    score = {}
    for key, value in imgX.items():
        score[key] = mse(np.array(T_img), np.array(imgX[key]))
        ans = 0
        try:
            ans = min(score, key=score.get)
        except ValueError:
            print(score)
    for key, value in score.items():
        print(key + ' ' + str(value))
    return ans

def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err


def main():
    return -1

imgA = np.array(img['A'])
imgB = np.array(img['B'])
imgC = np.array(img['C'])
threshold = 200

t_ac = get_mse_transform(imgA, imgC, angel_set, transpose_set)
ind_ac = get_least_index(t_ac,threshold)
Tac_b_img = perform_transform(img['B'],ind_ac,angel_set,transpose_set)
answer2 = test_transform_X(imgX, Tac_b_img)
print(answer2)

t_ab = get_mse_transform(imgA, imgB, angel_set, transpose_set)
ind_ab = get_least_index(t_ab,threshold)
Tab_c_img = perform_transform(img['C'],ind_ab,angel_set,transpose_set)
answer1 = test_transform_X(imgX, Tab_c_img)
print(answer1)




