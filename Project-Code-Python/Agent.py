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


        # construct dictionary for images.
        for key, value in problem.figures.items():
            if problem.problemSetName[-1] == 'B':
                if key.isalpha() == True:
                    image_array = Image.open(value.visualFilename)
                    # image_array = np.array(Image.open(images[key]))
                    img[key] = image_array
                    # img[key] = img_threshold(image_array, 128)
                # else key.isdigit() ==True:
                else:
                    imgX[key] = Image.open(value.visualFilename)
                    # imgX[key] = np.array(Image.open(images[key]))
                    # imgX[key] = img_threshold(np.array(Image.open(images[key])), 128)
            else: exit()

        def mse(imageA, imageB):  # eat up 2 arrays.
            # the 'Mean Squared Error' between the two images is the
            # sum of the squared difference between the two images;
            # NOTE: the two images must have the same dimension
            err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
            err /= float(imageA.shape[0] * imageA.shape[1])

            # return the MSE, the lower the error, the more "similar"
            # the two images are
            return err

        def self_symmetric(img, threshold):  # eat up a image object.
            ans = 0
            symmetry_set = [Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM, Image.TRANSPOSE]
            for trans in symmetry_set:
                if mse(np.array(img), np.array(img.transpose(trans))) < threshold:
                    ans = 1
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

        answer = 0
        # Rule No.1: Combine and find a self-symmetry property. 4/12 of basic and 4/12 of test are correct.
        for key,value in imgX.items():
            c = combine_figures(img, imgX[key])
            if self_symmetric(c, 400) == 1:
                answer = key

        #Rule No.2 simple transformation from A to B or A to C
        angel_set = np.array([0])
        transpose_set = [-1, Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM, Image.TRANSPOSE, Image.ROTATE_90,
                         Image.ROTATE_180,
                         Image.ROTATE_270]

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

        if answer == 1:  # no symmetry was found
            return int(answer)
        else:
            threshold1 = 400
            threshold2 = 2000
            t_ac = get_mse_transform(np.array(img['A']), np.array(img['C']), angel_set,
                                     transpose_set)  # eat up np.arrays
            ind_ac = get_similar_index(t_ac, threshold1)
            trans_ac = []  # store possible transformations
            for i, j in enumerate(ind_ac[0]):
                trans_ind = (ind_ac[0][i], ind_ac[1][i])
                Tac_b_img = perform_transform(np.array(img['B']), trans_ind, angel_set, transpose_set)
                an = test_transform_X(imgX, Tac_b_img, threshold2)
                trans_ac.append(an)  # eat up images

            if len(trans_ac)!=0:
                answer = trans_ac[0]

        return int(answer)
