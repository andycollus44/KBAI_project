from PIL import Image
import numpy as np
import os,glob

home = 'C:/Users/s2235/PycharmProjects/KBAI_project/Project-Code-Python/'
problemset = 'Basic Problems B/Basic Problem B-01/'
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
    im_trans_1 = 100000  # initialize bigger value of for mse comparison
    im_trans_2 = 100000
    for key,value in imgX.items():
        for angel in angel_set:
            for trans in transpose_set:
                im = Image.fromarray(img['A'])  # im is an image, img is an array
                try:
                    im.transpose(trans).rotate(angel)
                except ValueError:
                    im.rotate(angel)
                tmp1 = np.add(np.subtract(img['C'], np.array(im)), img['B'])  # C-A+B
                im_trans_1 = np.minimum(mse(tmp1,imgX[key]), im_trans_1)            #       smallest mse
                tmp2 = np.add(np.subtract(img['B'], np.array(im)), img['C'])  # B-A+C
                im_trans_2 = np.minimum(mse(tmp2,imgX[key]), im_trans_2)
        score1[key] = im_trans_1
        score2[key] = im_trans_2
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
