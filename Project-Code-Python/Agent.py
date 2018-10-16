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
            if problem.problemSetName[-1] == 'C':
                if key.isalpha() == True:
                    img[key] = Image.open(value.visualFilename)
                # else key.isdigit() ==True:
                else:
                    imgX[key] = Image.open(value.visualFilename)
            else: exit()                                                    # only test problem set B.

        def self_symmetric(img, threshold):  # eat up a image object.
            ans = False
            symmetry_set = [Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM]
            for trans in symmetry_set:
                if mse(np.array(img), np.array(img.transpose(trans))) < threshold:  # if it's symmetric to itself
                    ans = True
                    break
            return ans

        def combine_figures(img_dic, imgX):  # update for 3x3 problem
            w, h = img_dic['A'].size
            combined_img = Image.new('RGBA', (w * 3, h * 3))
            combined_img.paste(img_dic['A'], (0, 0))
            combined_img.paste(img_dic['B'], (w, 0))
            combined_img.paste(img_dic['C'], (2 * w, 0))
            combined_img.paste(img_dic['D'], (0, h))
            combined_img.paste(img_dic['E'], (w, h))
            combined_img.paste(img_dic['F'], (2 * w, h))
            combined_img.paste(img_dic['G'], (0, 2 * h))
            combined_img.paste(img_dic['H'], (w, 2 * h))
            combined_img.paste(imgX, (2 * w, 2 * h))
            return combined_img




        # Start testing here!
        answer = 0
        # Rule 1: The combined figure is symmetric to itself.
        for key,value in imgX.items():
            if self_symmetric(combine_figures(img, imgX[key]),3000)==True:      # the threshold should be < 3000
                answer = key


        return int(answer)
