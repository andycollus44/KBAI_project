# Your Agent for solving Raven's Progressive Matrices. You MUST modify this file.
#
# You may also create and submit new files in addition to modifying this file.
#
# Make sure your file retains methods with the signatures:
# def __init__(self)
# def Solve(self,problem)
#
# These methods will be necessary for the project's main method to run.

# Install Pillow and uncomment this line to access image processing.
# from PIL import Image
# import numpy

class Agent:
    # The default constructor for your Agent. Make sure to execute any
    # processing necessary before your Agent starts solving problems here.
    #
    # Do not add any variables to this signature; they will not be used by
    # main().

    def __init__(self):
        from PIL import Image

    # The primary method for solving incoming Raven's Progressive Matrices.
    # For each problem, your Agent's Solve() method will be called. At the
    # conclusion of Solve(), your Agent should return an int representing its
    # answer to the question: 1, 2, 3, 4, 5, or 6. Strings of these ints 
    # are also the Names of the individual RavensFigures, obtained through
    # RavensFigure.getName(). Return a negative number to skip a problem.
    #
    # Make sure to return your answer *as an integer* at the end of Solve().
    # Returning your answer as a string may cause your program to crash.
    def Solve(self, problem):
        from PIL import Image
        import numpy as np
        img = {}  # clues
        imgX = {}  # solutions
        # angel_set = np.array([0, 30, 45, 60, 90])  # remember to fill the background as white.
        angel_set = np.array([0])
        transpose_set = [-1,Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM, Image.TRANSPOSE,Image.ROTATE_90, Image.ROTATE_180,Image.ROTATE_270]

        def mse(imageA, imageB):
            # the 'Mean Squared Error' between the two images is the
            # sum of the squared difference between the two images;
            # NOTE: the two images must have the same dimension
            err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
            err /= float(imageA.shape[0] * imageA.shape[1])

            # return the MSE, the lower the error, the more "similar"
            # the two images are
            return err

        # construct dictionary for images.
        for key, value in problem.figures.items():
            if problem.problemSetName[-1] == 'B':
                if key.isalpha() == True:
                    img[key] = Image.open(value.visualFilename)
                # else key.isdigit() ==True:
                else:
                    imgX[key] = Image.open(value.visualFilename)

        def self_symmetric(img, threshold):  # eat up a image object.
            ans = False
            symmetry_set = [Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM]
            for trans in symmetry_set:
                if mse(np.array(img), np.array(img.transpose(trans))) < threshold:  # if it's symmetric to itself
                    ans = True
                    break
            return ans

        def combine_figures(img_dic, imgX):
            w, h = img_dic['A'].size
            combined_img = Image.new('RGBA', (w * 2, h * 2))
            combined_img.paste(img_dic['A'], (0, 0))
            combined_img.paste(img_dic['B'], (w, 0))
            combined_img.paste(img_dic['C'], (0, h))
            combined_img.paste(imgX, (w, h))
            return combined_img

        def test_transform_X(imgX, T_img, threshold):  # imgX dic of X.png, return the answer by comparing the TC and X
            score = []
            ind = []
            ans = 0
            for key, value in imgX.items():
                score.append(mse(np.array(T_img), np.array(imgX[key])))
                ind.append(key)

            if min(score) < threshold:
                ans = ind[np.argmin(score)]
            return ans

        def perform_transform(img, ind_tuple, angel_set, transpose_set):  # return an Image
            im = Image.fromarray(img)  # im is an image, img is an array
            try:  # allow no transpose
                t_img = im.transpose(transpose_set[ind_tuple[1]]).rotate(angel_set[ind_tuple[0]])
            except ValueError:
                t_img = im.rotate(angel_set[ind_tuple[0]])
            return t_img

        def get_mse_transform(imgA, imgB, angel_set,
                              transpose_set):  # transform 2 arrays and find mse follow the transformations
            trans_array = np.zeros((len(angel_set), len(transpose_set)))  # store the  tmpTABs
            for i, angel in enumerate(angel_set):
                for j, trans in enumerate(transpose_set):
                    im = Image.fromarray(imgA)  # im is an image, img is an array
                    try:  # allow no transpose
                        im = im.transpose(trans).rotate(angel)
                    except ValueError:
                        im = im.rotate(angel)

                    tmpTAB = mse(np.array(im), np.array(imgB))  # T(A)-B
                    trans_array[i][j] = tmpTAB

            return trans_array

        def get_similar_index(trans_array, threshold):
            ind = np.where(trans_array <= threshold)  # return all the values!
            return ind

        def get_change_point(row, threshold):
            thres_row = row.copy()
            thres_row[row > threshold] = 1
            thres_row[row <= threshold] = 0
            change_point = {0: row[0]}
            for index, pixel in enumerate(row[0:-2]):
                if pixel != row[index + 1]:
                    change_point[index + 1] = row[index + 1]
            return change_point  # return a dic of changed points.

        def squeeze_row(row, threshold):  # 00100100 to 01010, eat up a row of np.array
            thres_row = row.copy()
            thres_row[row > threshold] = 1
            thres_row[row <= threshold] = 0
            squeezed_row = [thres_row[0]]
            ind = [0]
            for index, pixel in enumerate(thres_row[0:-2]):
                if pixel != thres_row[index + 1]:
                    squeezed_row.append(thres_row[index + 1])
                    ind.append(index + 1)
            return (ind, squeezed_row)

        def fill_shape(img, threshold):  # fill single shape, eat up a image object
            im_array = np.array(img)
            for i, row in enumerate(im_array):
                #     ind_dic = get_change_point(row[:, 0], 128)

                sq_row = squeeze_row(row[:, 0], threshold)[1]  # find number of 0 to determine the layers.
                sq_ind = squeeze_row(row[:, 0], threshold)[0]
                if len(sq_ind) > 3:
                    fill_seg_ind = np.array(sq_ind)[np.array(sq_row) == 0]  # find black boundary
                    # fill_seg_ind = [i for i,ind in enumerate(sq_row) if ind==0]
                    row[fill_seg_ind[0]:fill_seg_ind[1],
                    0:2] = 0  # fill space between the boundary to black, first 3 channels
                    im_array[i] = row
            return Image.fromarray(im_array)

        def fill_compare(img, imgX, threshold):
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
                    if answer == 0:
                        if mse(np.array(img['A'])[:, :, 0], np.array(fill_shape(img['C'], 20))[:, :, 0]) < threshold:
                            for key, value in imgX.items():
                                im_ch1 = np.array(img['B'])[:, :, 0]
                                X_ch1 = np.array(fill_shape(value, 20))[:, :, 0]
                                if mse(X_ch1, im_ch1) < threshold:
                                    answer = key

            return answer

        def transform_compare(img, imgX, threshold):

            t_ac = get_mse_transform(np.array(img['A']), np.array(img['C']), angel_set,
                                     transpose_set)  # eat up np.arrays
            ind_ac = get_similar_index(t_ac, threshold)
            trans_ac = []  # store possible transformations
            for i, j in enumerate(ind_ac[0]):
                trans_ind = (ind_ac[0][i], ind_ac[1][i])
                Tac_b_img = perform_transform(np.array(img['B']), trans_ind, angel_set, transpose_set)
                an = test_transform_X(imgX, Tac_b_img, threshold)
                trans_ac.append(an)  # eat up images
            if len(trans_ac) != 0:
                return trans_ac[0]
            else:
                t_ab = get_mse_transform(np.array(img['A']), np.array(img['B']), angel_set,
                                         transpose_set)  # eat up np.arrays
                ind_ab = get_similar_index(t_ab, threshold)
                trans_ab = []  # store possible transformations
                for i, j in enumerate(ind_ab[0]):
                    trans_ind = (ind_ab[0][i], ind_ab[1][i])
                    Tab_c_img = perform_transform(np.array(img['C']), trans_ind, angel_set, transpose_set)
                    an = test_transform_X(imgX, Tab_c_img, threshold)
                    trans_ab.append(an)
                if len(trans_ab)!=0:
                    return trans_ab[0]
                else: return 0


        answer = 0
        # Rule 1: The combined figure is symmetric to itself.
        for key,value in imgX.items():
            if self_symmetric(combine_figures(img, imgX[key]),3000)==True:
                answer = key

            if answer == 0:
                # Rule 2: fill a shape and compare
                answer = fill_compare(img,imgX,3000)

                if answer ==0:
                    #Rule 3: transform and compare
                    answer = transform_compare(img,imgX,2000)

        return int(answer)
