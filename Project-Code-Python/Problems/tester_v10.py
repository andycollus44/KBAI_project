# tester4 is simplified for basic B problems.
from PIL import Image
import numpy as np
import os, glob
import time

# home = 'C:/Users/s2235/PycharmProjects/KBAI_project/Project-Code-Python/'
home = 'd:/PycharmProjects/KBAI_project/Project-Code-Python/'
problemset = 'Basic Problems B/Basic Problem B-11/'
os.chdir(home + 'Problems/' + problemset)

images = {}
img = {}
imgX = {}


for filename in glob.glob('*.png'):
    if len(filename) == 5:
        images[filename[0]] = home + 'Problems/' + problemset + filename

# build dic of pics.
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
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err

# Rule No.1: Combine and find a self-symmetry property.
def self_symmetric (img,threshold):   # eat up a image object.
    ans = 0
    symmetry_set = [Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM]
    for trans in symmetry_set:
        if mse(np.array(img),np.array(img.transpose(trans)))<threshold:
            ans = 1
            break
    return ans

def combine_figures(img_dic,imgX):
    w,h = img_dic['A'].size
    combined_img = Image.new('RGBA', (w*2,h*2))
    combined_img.paste(img_dic['A'],(0,0))
    combined_img.paste(img_dic['B'], (w, 0))
    combined_img.paste(img_dic['C'], (0, h))
    combined_img.paste(imgX, (w, h))
    return combined_img


angel_set = np.array([0])
transpose_set = [-1, Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM, Image.TRANSPOSE, Image.ROTATE_90, Image.ROTATE_180,
                 Image.ROTATE_270]

def test_transform_X(imgX, T_img,threshold):  # imgX dic of X.png, return the answer by comparing the TC and X
    score = []
    ind = []
    ans = 0
    for key, value in imgX.items():
        score.append(mse(np.array(T_img), np.array(imgX[key])))
        ind.append(key)

    if min(score)<threshold:
        ans = ind[np.argmin(score)]
    return ans

def perform_transform(img,ind_tuple,angel_set,transpose_set):       #return an Image
    im = Image.fromarray(img)  # im is an image, img is an array
    try:  # allow no transpose
        t_img = im.transpose(transpose_set[ind_tuple[1]]).rotate(angel_set[ind_tuple[0]])
    except ValueError:
        t_img = im.rotate(angel_set[ind_tuple[0]])
    return t_img

def get_mse_transform(imgA, imgB, angel_set, transpose_set):     #transform 2 arrays and find mse follow the transformations
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

def get_similar_index(trans_array,threshold):
    ind = np.where(trans_array <=threshold)     #return all the values!
    return ind

c = combine_figures(img,imgX['5'])

ans = self_symmetric(c,400)

# print(a)
if ans ==1:       # no symmetry was found
    exit()
else:
    threshold1 = 400
    threshold2 = 2000
    t_ac = get_mse_transform(np.array(img['A']), np.array(img['C']), angel_set, transpose_set) # eat up np.arrays
    ind_ac = get_similar_index(t_ac, threshold1)
    trans_ac = [] # store possible transformations
    for i, j in enumerate(ind_ac[0]):
        trans_ind = (ind_ac[0][i], ind_ac[1][i])
        Tac_b_img = perform_transform(np.array(img['B']), trans_ind, angel_set, transpose_set)
        an = test_transform_X(imgX, Tac_b_img,threshold2)
        trans_ac.append(an)      # eat up images

def transform_compare(img,imgX,threshold):

    t_ac = get_mse_transform(np.array(img['A']), np.array(img['C']), angel_set, transpose_set) # eat up np.arrays
    ind_ac = get_similar_index(t_ac, threshold)
    trans_ac = [] # store possible transformations
    for i, j in enumerate(ind_ac[0]):
        trans_ind = (ind_ac[0][i], ind_ac[1][i])
        Tac_b_img = perform_transform(np.array(img['B']), trans_ind, angel_set, transpose_set)
        an = test_transform_X(imgX, Tac_b_img,threshold)
        trans_ac.append(an)      # eat up images
    if len(trans_ac)!=0:
        return trans_ac[0]
    else:
        t_ab = get_mse_transform(np.array(img['A']), np.array(img['B']), angel_set, transpose_set)  # eat up np.arrays
        ind_ab = get_similar_index(t_ab, threshold)
        trans_ab = []  # store possible transformations
        for i, j in enumerate(ind_ab[0]):
            trans_ind = (ind_ab[0][i], ind_ab[1][i])
            Tab_c_img = perform_transform(np.array(img['C']), trans_ind, angel_set, transpose_set)
            an = test_transform_X(imgX, Tab_c_img, threshold)
            trans_ab.append(an)
        return trans_ab[0]


# Rule 3 fill a ring shape. A-B or B-A

def get_change_point(row,threshold):
    thres_row = row.copy()
    thres_row[row> threshold] = 1
    thres_row[row<=threshold] = 0
    change_point = {0:row[0]}
    for index,pixel in enumerate(row[0:-2]):
        if pixel != row[index+1]:
            change_point[index+1] = row[index+1]
    return change_point                 # return a dic of changed points.

def squeeze_row(row,threshold): # 00100100 to 01010, eat up a row of np.array
    thres_row = row.copy()
    thres_row[row> threshold] = 1
    thres_row[row<=threshold] = 0
    squeezed_row = [thres_row[0]]
    ind = [0]
    for index,pixel in enumerate(thres_row[0:-2]):
        if pixel != thres_row[index+1]:
            squeezed_row.append(thres_row[index+1])
            ind.append(index+1)
    return (ind,squeezed_row)

def fill_shape(img,threshold):           # fill single shape, eat up a image object
    im_array = np.array(img)
    for i, row in enumerate(im_array):
        #     ind_dic = get_change_point(row[:, 0], 128)

        sq_row = squeeze_row(row[:, 0], threshold)[1]  # find number of 0 to determine the layers.
        sq_ind = squeeze_row(row[:, 0], threshold)[0]
        if len(sq_ind) > 3:
            fill_seg_ind = np.array(sq_ind)[np.array(sq_row) == 0]  # find black boundary
            # fill_seg_ind = [i for i,ind in enumerate(sq_row) if ind==0]
            row[fill_seg_ind[0]:fill_seg_ind[1], 0:2] = 0  # fill space between the boundary to black, first 3 channels
            im_array[i] = row
    return Image.fromarray(im_array)

def fill_compare(img,imgX,threshold):
    answer = 0
    if mse(np.array(img['B'])[:, :, 0], np.array(fill_shape(img['A'], 20))[:, :, 0]) < threshold:
        for key, value in imgX.items():
            X_ch1 = np.array(value)[:, :, 0]
            im_ch1 = np.array(fill_shape(img['C'], 20))[:, :, 0]
            if mse(X_ch1, im_ch1) < threshold:
                answer = key
    if answer == 0:
        if mse(np.array(img['A'])[:, :, 0], np.array(fill_shape(img['B'], 20))[:, :, 0]) < threshold:
            for key, value in imgX.items():
                im_ch1 = np.array(img['C'])[:, :, 0]
                X_ch1 = np.array(fill_shape(value, 20))[:, :, 0]
                if mse(X_ch1, im_ch1) < threshold:
                    answer = key
        if answer == 0:
            if mse(np.array(img['C'])[:, :, 0], np.array(fill_shape(img['A'], 20))[:, :, 0]) < threshold:
                for key, value in imgX.items():
                    X_ch1 = np.array(value)[:, :, 0]
                    im_ch1 = np.array(fill_shape(img['B'], 20))[:, :, 0]
                    if mse(X_ch1, im_ch1) < threshold:
                        answer = key
            if answer ==0:
                if mse(np.array(img['A'])[:, :, 0], np.array(fill_shape(img['C'], 20))[:, :, 0]) < threshold:
                    for key, value in imgX.items():
                        im_ch1 = np.array(img['B'])[:, :, 0]
                        X_ch1 = np.array(fill_shape(value, 20))[:, :, 0]
                        if mse(X_ch1, im_ch1) < threshold:
                            answer = key

    return answer

# print(fill_compare(img,imgX,2000))

def img_threshold(img, thres):      # better for subtraction
    img = np.array(img)
    img_tmp = img.copy()
    img_tmp[img > thres] = 10  # white to 1
    img_tmp[img <= thres] = 3  # black to 2
    return img_tmp

def img_subtract(imgA,imgB):        #
    diff = np.subtract(np.array(imgA, dtype='int8'), np.array(imgB, dtype='int8'))
    return diff


def diff_compare(img, imgX):
    # B-A = X-C
    score = []
    ind = []
    for key, value in imgX.items():
        BsubA = img_subtract(img_threshold(img['B'], 128), img_threshold(img['A'], 128))
        XsubC = img_subtract(img_threshold(imgX[key], 128), img_threshold(img['C'], 128))
        score.append(mse(BsubA, XsubC))
        ind.append(key)
    return ind[np.argmin(score)]

print(diff_compare(img,imgX))




def test_transform(imgX, img, angel_set, transpose_set):  # imgX dic of
    score1 = {}
    score2 = {}
    avg_score = {}

    for key, value in imgX.items():
        t_list1 = []  # t_list to store MSE and T to get all valid transpose from B-A
        t_list2 = []
        for angel in angel_set:
            for trans in transpose_set:
                im = Image.fromarray(img['A'])  # im is an image, img is an array
                try:  # allow no transpose
                    im = im.transpose(trans).rotate(angel)
                except ValueError:
                    im = im.rotate(angel)

                tmpTAB = np.subtract(np.array(im,dtype='int8'), np.array(img['B'],dtype='int8'))  # T(A)-B
                tmpTCX = np.subtract(np.array(img['C'],dtype='int8'), np.array(imgX[key],dtype='int8'))  # T(C)-X
                t1 = mse(tmpTAB, tmpTCX)  # the transposed image difference
                t_list1.append(t1)  # dic for all mse of differences, to get minimum
                tmpTAC = np.subtract(np.array(im,dtype='int8'), np.array(img['C'],dtype='int8'))  # T(A)-C
                tmpTBX = np.subtract(np.array(img['B'],dtype='int8'), np.array(imgX[key],dtype='int8'))  # T(C)-X
                t2 = mse(tmpTAC, tmpTBX)
                t_list2.append(t2)
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
    for key, value in avg_score.items():
        print(key + ' ' + str(value))
    return ans

