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


        # construct dictionary for images.
        for key, value in problem.figures.items():
            if problem.problemSetName[-1] == 'B':
                if key.isalpha() == True:
                    img[key] = np.array(Image.open(value.visualFilename))
                # else key.isdigit() ==True:
                else:
                    imgX[key] = np.array(Image.open(value.visualFilename))

        answer = test_transform(imgX, img, angel_set, transpose_set)

        return int(answer)
