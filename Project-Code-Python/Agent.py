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
        angel_set = np.array([0, 30, 45, 60, 90])  # remember to fill the background as white.
        transpose_set = [Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM, Image.TRANSPOSE]

        def test_transform(imgX, img, angel_set, transpose_set):  # imgX dic of
            score1 = {}
            score2 = {}
            avg_score = {}
            im_trans_1 = 10000  # initialize bigger value of for mse comparison
            im_trans_2 = 10000
            for key in imgX.items():
                for angel in angel_set:
                    for trans in transpose_set:
                        im = Image.fromarray(img['A'])  # im is an image, img is an array

                        im.transpose(trans).rotate(angel)
                        tmp1 = np.add(np.substract(img['C'], np.array(im)), img['B'])  # C-A+B
                        im_trans_1 = min(tmp1, im_trans_1)
                        tmp2 = np.add(np.substract(img['B'], np.array(im)), img['C'])  # B-A+C
                        im_trans_2 = min(tmp2, im_trans_2)
                score1[key] = mse(imgX[key], im_trans_1)
                score2[key] = mse(imgX[key], im_trans_2)
            for key, value in score1:
                avg_score[key] = float((score1[key] + score2[key]) / 2)
            return max(avg_score, key=avg_score.get)

        def mse(imageA, imageB):
            # the 'Mean Squared Error' between the two images is the
            # sum of the squared difference between the two images;
            # NOTE: the two images must have the same dimension
            err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
            err /= float(imageA.shape[0] * imageA.shape[1])

            # return the MSE, the lower the error, the more "similar"
            # the two images are
            return err



        for key, value in problem.figures.items():
            if problem.problemSetName[-1] == 'B':
                if key.isalpha == True:
                    img[key] = np.array(Image.open(value.visualFilename))
                elif key.isdigit ==True:
                    imgX[key] = np.array(Image.open(value.visualFilename))

                answer = test_transform(imgX, img, angel_set, transpose_set)

        return answer
