# tester for T(B)-A and T(C)-X, get the one with minimum difference.

from PIL import Image
import numpy as np
import os,glob
import time

home = 'C:/Users/s2235/PycharmProjects/KBAI_project/Project-Code-Python/'
problemset = 'Basic Problems B/Basic Problem B-05/'
os.chdir(home+'Problems/'+problemset)

images = {}
img = {}
imgX= {}


for filename in glob.glob('*.png'):
    if len(filename)==5:
        images[filename[0]]=home+'Problems/'+problemset+filename

for key, value in images.items():

    if key.isalpha() == True:
        img[key] = np.array(Image.open(images[key]))
    # else key.isdigit() ==True:
    else:
        imgX[key] = np.array(Image.open(images[key]))


angel_set = np.array([0])
transpose_set = [-1,Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM, Image.TRANSPOSE,Image.ROTATE_90, Image.ROTATE_180,Image.ROTATE_270]

def test_transform(imgX, img, angel_set, transpose_set):  # imgX dic of
    score1 = {}
    score2 = {}
    avg_score = {}

    for key,value in imgX.items():
        t_list1 = np.multiply(np.ones(100),500000)  # t_list to store MSE and T to get all valid transpose from B-A
        t_list2 = np.multiply(np.ones(100),500000)
        for angel in angel_set:
            for i,trans in enumerate(transpose_set):
                im = Image.fromarray(img['A'])  # im is an image, img is an array
                try:            # allow no transpose
                    im = im.transpose(trans).rotate(angel)
                except ValueError:
                    im = im.rotate(angel)
                tmpTAB = np.subtract(np.array(im),img['B'])  # T(A)-B
                tmpTCX = np.subtract(img['C'],imgX[key])         # T(C)-X
                t1 = mse(tmpTAB,tmpTCX)           #       the transposed image difference
                t_list1[i]=t1                       #dic for all mse, to get minimum
                tmpTAC = np.subtract(np.array(im), img['C'])  # T(A)-C
                tmpTBX = np.subtract(img['B'], imgX[key])  # T(C)-X
                t2 = mse(tmpTAC,tmpTBX)
                t_list2[i]=t2
                # check points
                # print(key+' C-A+B '+str(im_trans_1))
                # Image.fromarray(tmp1).show()
                # pause()
                # print(key + ' B-A+C '+str(im_trans_2))
                # Image.fromarray(tmp2).show()
                # pause()
        score1[key] = min(t_list1)
        score2[key] = min(t_list2)

    ans = 0
    for key, value in score1.items():
        avg_score[key] = float((score1[key] + score2[key]) / 2)

        try:
            ans = min(avg_score, key=avg_score.get)
        except ValueError:
            print(avg_score)
    for key,value in avg_score.items():
        print(key+' '+str(value))
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

answer = test_transform(imgX, img, angel_set, transpose_set)
print(answer)