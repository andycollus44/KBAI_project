from PIL import Image
import os
import numpy as np

os.chdir('Project-Code-Python/Problems/Basic Problems B/Basic Problem B-01/')
img = Image.open('1.png')
#width,height = img.size

pix_matrix = np.zeros(img.size)

# pixels = list(img.getdata())
pix = np.array(img)
width,height,chs = pix.shape
# we will iterate through channels. First we just deal with 1st channel.
ch1 = pix[:,:,1]

# pad the matrix to 3n
if np.mod(width,3)!=0:
    w_pad = 3-np.mod(width,3)
else:
    w_pad = 0

if np.mod(height,3)!=0:
    h_pad = 3-np.mod(height,3)
else:
    h_pad = 0

ch1_padded = np.pad(ch1,((0,w_pad),(0,h_pad)),'constant',constant_values = (0,255))
#iterate every 3by3 matrix
w_range = range(width+w_pad)
h_range = range(height+h_pad)
for w in w_range[2::2]:
    


#pix.shape
#arr2im = Image.fromarray(pix)
#mat = pix.reshape(width,height)

# For set B, detect close shapes as objects.
def close_shape_detect(img):
    pixel_matrix = np.zeros(img.size)