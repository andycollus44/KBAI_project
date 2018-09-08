from PIL import Image
import os
import numpy as np

os.chdir('Project-Code-Python/Problems/Basic Problems B/Basic Problem B-01/')
img = Image.open('1.png')
#width,height = img.size



# pixels = list(img.getdata())
im_array = np.array(img)
pix = im_array
global width,height,chs
width,height,chs = im_array.shape
pix[im_array > 10] = 255
pix[im_array <= 10] = 0

pix_matrix = np.zeros((width*height,width*height),dtype=bool) # the relation matrix of pixels, from 0

# we will iterate through channels. First we just deal with 1st channel.
ch1 = pix[:,:,1]

# pad the matrix to 3n
# if np.mod(width,3)!=0:
#     w_pad = 3-np.mod(width,3)
# else:
#     w_pad = 0
#
# if np.mod(height,3)!=0:
#     h_pad = 3-np.mod(height,3)
# else:
#     h_pad = 0
#
# ch1_padded = np.pad(ch1,((0,w_pad),(0,h_pad)),'constant',constant_values = (0,255))
# #iterate every 3by3 matrix
# w_range = range(width+w_pad)
# h_range = range(height+h_pad)
# for w in w_range[2::2]:
#     for h in h_range[2::2]:
#         mat_3=np.array(ch1_padded[h-1:h+1,w-1:w+1])
#         if np.sum(mat_3)<8*255:
#             np.argwhere(mat_3 == 0)

#iterate from 2nd to n-1 th
w_range = range(1,width-2)
h_range = range(1,height-2)
for w in w_range:
    for h in h_range:
        mat_3=np.array(ch1[h-1:h+2,w-1:w+2])
        if np.sum(mat_3)<8*255:
            pixel_touch = np.argwhere(mat_3 == 0)
            for i in pixel_touch:
                pix_matrix[(h*width+w),((h-1+i[0])*width+(w-1)+i[1])]=1         #y*width+x,
                pix_matrix[((h - 1 + i[0]) * width + (w - 1) + i[1]),(h * width + w)] = 1

#reconstruct object from relation matrix
global reconst_img
reconst_img = np.zeros((width,height))

def get_relation_list(pix_mat,p):
    relation = pix_matrix[:, p]
    ind = np.argwhere(relation == 1)
    return ind

def extract_relation(pix_mat,p):
    ind = get_relation_list(pix_mat,p)
    fill_related_pix(p,ind)
    for q in ind:
        extract_relation(pix_mat,q[0])

def fill(x):
    w = np.mod(x, width)
    h = int((x - w) / width)
    reconst_img[h, w] = 255

#m ind of relation matrix, n list of array of related ind.
def fill_related_pix(m,n):
    fill(m)
    for i in n:
        fill(i[0])


p = np.argwhere(pix_matrix==1)[1][0]
extract_relation(pix_matrix,p)

#pix.shape
#arr2im = Image.fromarray(pix)
#mat = pix.reshape(width,height)

# For set B, detect close shapes as objects.
def close_shape_detect(img):
    pixel_matrix = np.zeros(img.size)