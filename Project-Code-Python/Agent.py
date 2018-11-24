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
        from PIL import ImageFilter
        import numpy as np
        img = {}  # clues
        imgX = {}  # solutions
        # angel_set = np.array([0, 30, 45, 60, 90])  # remember to fill the background as white.
        angel_set = np.array([0])
        transpose_set = [-1,Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM, Image.TRANSPOSE,Image.ROTATE_90, Image.ROTATE_180,Image.ROTATE_270]

        def mse(imageA, imageB):  # eat up 2 arrays.
            # the 'Mean Squared Error' between the two images is the
            # sum of the squared difference between the two images;
            # NOTE: the two images must have the same dimension
            imgarrayA = np.array(imageA.filter(ImageFilter.GaussianBlur(radius=10)))
            imgarrayB = np.array(imageB.filter(ImageFilter.GaussianBlur(radius=10)))
            err = np.sum((imgarrayA.astype("float") - imgarrayB.astype("float")) ** 2)
            err /= float(imgarrayA.shape[0] * imgarrayA.shape[1])

            # return the MSE, the lower the error, the more "similar"
            # the two images are
            return err

        def mset(imageA, imageB, thres):  # eat up 2 arrays.
            # the 'Mean Squared Error' between the two images is the
            # sum of the squared difference between the two images;
            # NOTE: the two images must have the same dimension
            arrayA = np.array(imageA.filter(ImageFilter.GaussianBlur(radius=10)))
            arrayB = np.array(imageB.filter(ImageFilter.GaussianBlur(radius=10)))
            imgarrayA = arrayA.copy()
            imgarrayB = arrayB.copy()
            imgarrayA[arrayA > thres] = 1
            imgarrayA[arrayA < thres] = 0
            imgarrayB[arrayB > thres] = 1
            imgarrayB[arrayB < thres] = 0
            err = np.sum((imgarrayA.astype("float") - imgarrayB.astype("float")) ** 2)
            err /= float(imgarrayA.shape[0] * imgarrayA.shape[1])

            # return the MSE, the lower the error, the more "similar"
            # the two images are
            return err

                                                        # only test problem set B.

        def self_symmetric(img, threshold):  # eat up a image object.
            ans = 0
            symmetry_set = [Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM]
            for trans in symmetry_set:
                if mse(img, img.transpose(trans)) < threshold:
                    ans = 1
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

        def dark_ratio(img, thres=128):
            imgarray = np.array(img)
            img_tmp = imgarray.copy()
            img_tmp[imgarray > thres] = 0  # white to 0
            img_tmp[imgarray <= thres] = 1  # black to 1
            return np.sum(img_tmp) / (img.size[0] * img.size[1])

        def dark_center(img, thres):
            # add the rows and average them.
            imgarray = np.array(img)
            img_tmp = imgarray.copy()
            img_tmp[imgarray > thres] = 0  # white to 0
            img_tmp[imgarray <= thres] = 1  # black to 1
            rows = np.sum(np.sum((img_tmp), axis=0), axis=1)
            cols = np.sum(np.sum((img_tmp), axis=1), axis=1)
            return (weighted_center(rows), weighted_center(cols))

        def weighted_center(row_array):  # pass a list
            sum = 0
            row_list = list(row_array)
            for i in range(len(row_list)):
                sum = sum + row_list[i] * (i + 1)
            return sum / np.sum(row_list)

        def figure_sum(img_dict):
            w, h = img_dict['A'].size
            sum_img = np.zeros((w, h, 4))
            for key, value in img_dict.items():
                sum_img = sum_img + np.array(value)
            return sum_img

        def calc_diff(list, value):
            list = np.array(list)
            value = np.array(value)
            exp1 = 2 * list[5] - list[2]
            exp2 = 2 * list[7] - list[6]
            exp3 = 2 * list[4] - list[0]
            return (value - exp1) ** 2 + (value - exp2) ** 2 + (value - exp3) ** 2

        def check_dark_increase(list,incre_factor):      # check if there's an increment in dark_ratio
            if np.abs(int(list[1]) - int(list[0])) < incre_factor * int(list[0]):
                return False
            else:
                return True

        def test_dark_ratio(img, imgX, thres):
            dark_list = []
            dark_center_list = []
            # # keys = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
            # keys = img.keys()
            # for k in sorted(img.iterkeys()):
            #     dark_list.append(dark_ratio(img[k], thres))             # put dark ratio values into a list
            #     dark_center_list.append(dark_center(img[k], thres))
            for key, value in sorted(img.items()):
                dark_list.append(dark_ratio(img[key], thres))  # put dark ratio values into a list
                dark_center_list.append(dark_center(img[key], thres))

            # if len(dark_list) == 0:
            #     print('empty dark list')
            # # elif check_dark_increase(dark_list, 0.1) == False:
            # #     return 0
            min_center = 1000
            min_diff = 10000000000
            ans = 0
            for key, value in imgX.items():

                diff = calc_diff(dark_list, dark_ratio(value, thres))
                center_dff = calc_diff(dark_center_list, dark_center(value, thres))
                if type(diff) == type((1, 1)):
                    diff = np.sum(diff)
                if diff < min_diff:
                    min_diff = diff
                    ans = int(key)
            return ans

        def test_horiz_switch(img,imgX,thres):
            ans = 0
            for key, value in imgX.items():
                if mse(horiz_switch(value),img['G'])<thres:
                    ans = key
            return ans


        def horiz_switch(img):

            imgarray = np.array(img)
            w, h, c = imgarray.shape
            img_tmp = imgarray.copy()
            img_tmp[:, 0:int(h / 2 - 1), :] = imgarray[:, int(h / 2):-1, :]
            img_tmp[:, int(h / 2):-1:] = imgarray[:, 0:int(h / 2) - 1, :]

            return Image.fromarray(img_tmp)

        def rolling_similarity(img, imgX, thres):
            answer = 0
            if mset(img['A'], img['E'], 128) < thres and mset(img['B'], img['F'],128) < thres and mset(img['C'],
                                                                                                 img['D'],128) < thres:

                for key, value in imgX.items():
                    if mse(value, img['E']) < thres:
                        answer = int(key)
            return answer

        def row_sub_similarity(img,imgX,thre1,thre2,thre3):
            answer = 0
            row_diff_sim = mse_array(img_subtract(img_subtract(img['A'], img['B']), img['C']),
                                     img_subtract(img_subtract(img['D'], img['E']), img['F']))
            if row_diff_sim < thre1:
                for key, value in imgX.items():
                    add_rule = mse(value, img['C'])
                    if mse_array(img_subtract(img_subtract(img['A'], img['B']), img['C']),
                                 img_subtract(img_subtract(img['G'], img['H']), value)) < thre2 and add_rule < thre3:
                        answer = key
            return answer

        def img_subtract(imgA, imgB):  #
            diff = np.subtract(np.array(imgA, dtype='int8'), np.array(imgB, dtype='int8'))
            return diff

        def mse_array(imgarrayA, imgarrayB):  # eat up 2 arrays.
            # the 'Mean Squared Error' between the two images is the
            # sum of the squared difference between the two images;
            # NOTE: the two images must have the same dimension
            # imgarrayA = array_blur(imgarrayA)
            # imgarrayB = array_blur(imgarrayB)
            err = np.sum((imgarrayA.astype("float") - imgarrayB.astype("float")) ** 2)
            err /= float(imgarrayA.shape[0] * imgarrayA.shape[1])

            # return the MSE, the lower the error, the more "similar"
            # the two images are
            return err

        def IPR(imageA, imageB, thres=128):
            arrayA = np.array(imageA.filter(ImageFilter.GaussianBlur(radius=2)))
            arrayB = np.array(imageB.filter(ImageFilter.GaussianBlur(radius=2)))
            imgarrayA = arrayA.copy()
            imgarrayB = arrayB.copy()
            imgarrayA[arrayA > thres] = 1
            imgarrayA[arrayA < thres] = 0
            imgarrayB[arrayB > thres] = 1
            imgarrayB[arrayB < thres] = 0
            sum = (imgarrayA + imgarrayB)
            both_white = np.count_nonzero(sum == 2)
            one_dark = np.count_nonzero(sum == 1)
            both_dark = np.count_nonzero(sum == 0)
            return (both_dark, one_dark, both_white)

        def IPR_ratio(imageA, imageB):
            (both_dark, one_dark, both_white) = IPR(imageA, imageB)
            sum = both_dark + one_dark + both_white
            return (both_dark / sum, one_dark / sum)

        def fig_sim(imageA, imageB, alpha=2):
            (both_dark, one_dark, both_white) = IPR(imageA, imageB)
            try:
                return both_dark / (both_dark + alpha * one_dark)
            except ZeroDivisionError as error:
                print(problem.problemSetName)
                return 0

        def calc_IPR_diff(img, img_fromX):
            return np.abs(fig_sim(img['C'], img['F']) - fig_sim(img_fromX, img['F'])) + np.abs(
                fig_sim(img['G'], img['H']) - fig_sim(img_fromX, img['H']) + np.abs(
                    fig_sim(img['A'], img['E']) - fig_sim(img_fromX, img['E'])))

        def calc_DPR_diff(img, img_fromX):
            return np.abs(np.abs(dark_ratio(img['C']) - dark_ratio(img['F'])) - np.abs(
                dark_ratio(img_fromX) - dark_ratio(img['F']))) + np.abs(
                np.abs(dark_ratio(img['G']) - dark_ratio(img['H'])) - np.abs(
                    dark_ratio(img_fromX) - dark_ratio(img['H']))) + np.abs(
                np.abs(dark_ratio(img['A']) - dark_ratio(img['E'])) - np.abs(
                    dark_ratio(img_fromX) - dark_ratio(img['E'])))

        def test_DPR_IPR(img, imgX):
            DPR_list = []
            IPR_list = []
            # # keys = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
            # keys = img.keys()
            # for k in sorted(img.iterkeys()):
            #     dark_list.append(dark_ratio(img[k], thres))             # put dark ratio values into a list
            #     dark_center_list.append(dark_center(img[k], thres))
            for key, value in sorted(imgX.items()):
                DPR_list.append(calc_DPR_diff(img, value))  # put dark ratio values into a list
                IPR_list.append(calc_IPR_diff(img, value))

            return (DPR_list, IPR_list)

        def rank_list(DPR_list):
            seq = sorted(DPR_list)
            DPR_rank = [seq.index(v) for v in DPR_list]
            return DPR_rank

        def rank_vote_answers(DPR_rank, IPR_rank):
            votes = [sum(x) for x in zip(DPR_rank, IPR_rank)]
            return votes.index(min(votes)) + 1

        def sum_vote_answers(DPR_list, IPR_list, alpha=0.7):                                       # thres!!!
            aDPR_list = [alpha * x for x in DPR_list]
            votes = [sum(x) for x in zip(aDPR_list, IPR_list)]
            return votes.index(min(votes)) + 1

        def dot_sum_vote_answers(DPR_list, IPR_list, alpha=1):  # thres!!!
            dl = np.array(DPR_list)
            il = np.array(IPR_list)
            votes = dl * il / (dl + il)
            votes = list(votes)
            return votes.index(min(votes)) + 1

        # construct dictionary for Set D images.
        for key, value in problem.figures.items():
            if problem.problemSetName[-1] == 'D' or problem.problemSetName[-1] == 'E':
                if key.isalpha() == True:
                    img[key] = Image.open(value.visualFilename).convert('L')
                # else key.isdigit() ==True:
                else:
                    imgX[key] = Image.open(value.visualFilename).convert('L')
            else:
                continue
                # Start testing here!
        answer = 0

        # select Set D, E:
        # select Set D, E:
        # select Set D, E:
        if problem.problemSetName[-1] == 'D' or problem.problemSetName[-1] == 'E':
            # Rule 1: The combined figure is symmetric to itself.
            for key,value in imgX.items():
                if self_symmetric(combine_figures(img, imgX[key]),100)==True:      # the threshold should be < 3000
                    answer = key

            # Rule 2: The rolling similarity.

            # if answer == 0:
            #     answer = rolling_similarity(img, imgX, 40)
            #
            # if answer == 0:
            #     answer = row_sub_similarity(img,imgX,700,100,1800)
            # if answer == 0:
            #     answer = test_horiz_switch(img,imgX,2000)
            # Rule 3: DPR and IPR
            if answer == 0:
                (DPR_list, IPR_list) = test_DPR_IPR(img, imgX)
                # answer = sum_vote_answers(DPR_list, IPR_list)
                answer = sum_vote_answers(rank_list(DPR_list), rank_list(IPR_list))
                # answer = dot_sum_vote_answers(rank_list(DPR_list), rank_list(IPR_list))
            return int(answer)

        elif problem.problemSetName[-1] == 'E':

            return 0

        else:
            return 0