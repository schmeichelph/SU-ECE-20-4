#
#
# Thursday 4/16/20
# Phil messing with add_template and Mask R-CNN mask
# 
# Currently reshapes the mask to the same size that is used in match(),
# but old MATLAB template is still used.
#
# Next step is to write the mask as an image with OpenCV stuff then use rec_list.add_template().
# Pretty sure manual_roi() will help with this
# 
#


"""
Authors: Ross Pitman, Jack Gularte, Devin DeWitt
File: recognition.py
Date Last Modified: April 24th, 2019
For: Panthera Organization
Purpose: Use computer vision to help determine the number of individual cats in
            a database.
Output: 'score_matrix.csv'; A matrix of similarity scores between each two images
            within the database
To Run: python recogntion.py -work_directory "path_to_work_directory"
"""

# EXAMPLE USAGES
################################################################################
# (if work directory is the directory the file resides in):
#       python recognition_new.py

# (if work directory is somewhere else you need to specify where):
#       python recognition_new.py -work_directory "c:/Users/Jack Gualrte/Desktop"
########################### END USAGES #########################################

# NOTES
################################################################################
"""
    * number of processes has to be less than the number of images in directory

    * if python2 'pip install opencv-contrib-python==3.3.1.11'
        'i dont think this script will work on python 2 anymore, some packages'
        ' are python 3 and up'
    * if python3 then 'pip install opencv-contrib-python==3.4.2.16'
"""
################################ END NOTES #####################################

# IMPORTS
################################################################################
# basic python
import os, sys
import time, datetime
from copy import deepcopy
import threading
import argparse
import glob
import re
import traceback
from pathlib import Path, PurePath
import json

# advanced
import cv2
import numpy as np

# mask-rcnn
#   import os
#   import sys
import random
import math
#   import re
#   import time
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import skimage.io
# Root directory of the project
#ROOT_DIR = os.path.abspath("../..")
ROOT_DIR = "Mask_RCNN-master"
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log

from samples.snow_leopard import snow_leopard

#%matplotlib inline 

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Path to Ballon trained weights
# You can download this file from the Releases page
# https://github.com/matterport/Mask_RCNN/releases
BALLON_WEIGHTS_PATH = "C:/Users/Phil/SU-ECE-19-7-master-MaskRCNN/Recognition/Mask_RCNN-master/logs/bottle20200221T0110"  # TODO: update this path
########################### END IMPORTS ########################################

# CLASS DEFINITION
################################################################################
class Recognition:
    'This class holds an image-template pair.'
    'Keeps pairs secure and together thorughout whole process and'
    'cuts down on code bloat. Also holds the image title split into its'
    'base characterisitics and the proper cat ID'
    def __init__(self):
        self.image_title = ""
        self.image = ""
        self.template_title = ""
        self.template = ""
        self.station = ""
        self.camera = ""
        self.date = ""
        self.time = ""
        self.cat_ID = ""

    def add_image(self, image_title, image):
        self.image_title = image_title
        self.image = image

    def add_template(self, template_title, template):
        self.template_title = template_title
        self.template = template

    def add_title_chars(self, station, camera, date, time):
        self.station = station
        self.camera = camera
        self.date = date
        self.time = time

    def add_cat_ID(self, cat):
        self.cat_ID = cat
########################### END CLASS DEFINITION ###############################

# FUNCTION DEFINITIONS (In Reverse Order of Call)
################################################################################
def check_matrix(rec_list, score_matrix):


    hit = 0
    hit_count = 0
    miss = 0
    miss_count = 0

    # traverse rows
    for row in range(score_matrix.shape[0]):

        primary_cat = rec_list[row].cat_ID
        primary_title = rec_list[row].image_title
        #print("Cat_ID: {0}; Image: {1}".format(primary_cat, primary_title))

        # traverse columns
        for column in range(score_matrix.shape[1]):
            # dont check the same image.
            if (row != column):
                secondary_cat = rec_list[column].cat_ID

                # Pull the 'hit' out of the score matrix
                if (primary_cat == secondary_cat):
                    hit = hit + score_matrix[row][column]
                    hit_count = hit_count + 1
                else:
                    miss = miss + score_matrix[row][column]
                    miss_count = miss_count + 1

    try:
        print("Hits: {0}; Avg. Hit: {1}".format(hit_count, hit/hit_count))
    except ZeroDivisionError:
        print("Hits: 0; Avg. Miss: 0")

    try:
        print("Misses: {0}; Avg. Miss: {1}".format(miss_count, miss/miss_count))
    except ZeroDivisionError:
        print("Misses: 0; Avg. Miss: 0")

################################################################################
def normalize_matrix(score_matrix):
    'Used to normalize the score matrix with respect to the highest value present'

    # get max score
    #max_matrix = score_matrix.max()

    # normalize
    #score_matrix = score_matrix / max_matrix

    # add identity matrix
    score_matrix = score_matrix + np.identity(len(score_matrix[1]))
    return score_matrix

################################################################################
def write_matches(kp_1, kp_2, good_points, primary_image, secondary_image, image_destination):
    'This function takes the output of the KNN matches and draws all the matching points'
    'between the two images. Writes the final product to the output directory'

    # parameters to pass into drawing function
    draw_params = dict(matchColor = (0,255,0),
                       flags = 0,
                       singlePointColor = (255,0,0))

    # draw the matches between two upper pictures and horizontally concatenate
    result = cv2.drawMatches(
        primary_image.image,
        kp_1,
        secondary_image.image,
        kp_2,
        good_points,
        None,
        **draw_params       ) # draw connections

    # use the cv2.drawMatches function to horizontally concatenate and draw no
    # matching lines. this creates the clean bottom images.
    result_clean = cv2.drawMatches(
        primary_image.image,
        None,
        secondary_image.image,
        None,
        None,
        None) # don't draw connections

    # This code is Ross Pitman. I dont exactly know what all the constants are but they
    # create the border and do more image preprocessing
    row, col= result.shape[:2]
    bottom = result[row-2:row, 0:col]
    bordersize = 5
    result_border = cv2.copyMakeBorder(
        result,
        top = bordersize,
        bottom = bordersize,
        left = bordersize,
        right = bordersize,
        borderType = cv2.BORDER_CONSTANT, value = [0,0,0] )

    # same as above
    row, col= result_clean.shape[:2]
    bottom = result_clean[row-2:row, 0:col]
    bordersize = 5
    result_clean_border = cv2.copyMakeBorder(
        result_clean,
        top = bordersize,
        bottom = bordersize,
        left = bordersize,
        right = bordersize,
        borderType = cv2.BORDER_CONSTANT, value = [0,0,0] )

    # vertically concatenate the matchesDrawn and clean images created before.
    result_vertical_concat = np.concatenate(
        (result_border, result_clean_border),
        axis = 0)

    # Take the image_destination and turn it into a Path object.
    # Then add the image names to the new path.
    # # TODO: For some reason it says the 'image_destination' object is
    #           a str type at this point in the program even though it is not.
    #           Look into why.
    image_path = image_destination.joinpath(str(len(good_points)) +
    "___" +
    re.sub(".jpg", "", os.path.basename(primary_image.image_title)) +
    "___" +
    re.sub(".jpg", ".JPG", os.path.basename(secondary_image.image_title))
    )

    # Finally, write the finished image to the output folder.
    cv2.imwrite(str(image_path), result_vertical_concat, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
################################################################################
def score_boosting(primary_image, secondary_image, good_points, parameters):
    'uses image characteristics to boost scores'
    score = len(good_points)

    if (primary_image.station == secondary_image.station):
        if (primary_image.camera == secondary_image.camera):
            if (primary_image.date == secondary_image.date):
                score = score * float(parameters['score_boosting']['date_score'])
            else:
                score = score * float(parameters['score_boosting']['camera_score'])
        else:
            score = score * float(parameters['score_boosting']['station_score'])

    return score
################################################################################
def match(primary_images, secondary_images, image_destination,
            start_i, score_matrix, write_threshold, parameters):
    'main function used for determining matches between two images.'
    'Finds the sift keypoints/descriptors and uses a KNN based matcher'
    'to filter out bad keypoints. Writes final output to score_matrix'

    # Begin loop on the primary imags to match. Due to multithreading of the
    # program this may not be the full set of images.
    for primary_count in range(len(primary_images)):

        print("Image of cat ", primary_count, ": ", primary_images[primary_count].image_title)
        print("\t\tMatching: " + os.path.basename(primary_images[primary_count].image_title) + "\n")

        # create mask from template and place over image to reduce ROI
        mask_1 = cv2.imread(primary_images[primary_count].template_title, -1) 
        print("dimensions mask_1: ", len(mask_1), len(mask_1[0]))
        print(mask_1)
        cv2.imshow("create mask from template", mask_1)
        mySift = cv2.xfeatures2d.SIFT_create()
        kp_1, desc_1 = mySift.detectAndCompute(primary_images[primary_count].image, mask_1)

        # paramter setup and create nearest nieghbor matcher
        index_params = dict(algorithm = 0, trees = 5)
        search_params = dict()
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        # Begin nested loopfor the images to be matched to. This secondary loop
        # will always iterate over the full dataset of images.
        for secondary_count in range(len(secondary_images)):

            # check if same image; if not, go into sophisticated matching
            if primary_images[primary_count].image_title != secondary_images[secondary_count].image_title:

                 # create mask from template
                 mask_2 = cv2.imread(secondary_images[secondary_count].template_title, -1)
                 kp_2, desc_2 = mySift.detectAndCompute(secondary_images[secondary_count].image, mask_2)

                 
                 # This section is for the presentation only, remove later
                 temp1 = cv2.resize(rec_list[primary_count].image, (960, 540))
                 temp2 = cv2.resize(rec_list[secondary_count].image, (960, 540))


                 horiz = np.hstack((temp1, temp2))
                 cv2.imshow("matched image to template", horiz)
                 cv2.waitKey(500)
                 cv2.destroyAllWindows()
                 
                 # end 

                 # check for matches
                 try:
                     # Check for similarities between pairs
                     matches = flann.knnMatch(desc_1, desc_2, k=2)

                     # Use Lowe's ratio test
                     good_points = []
                     for m, n in matches:
                         if m.distance < 0.7 * n.distance:
                             good_points.append(m)


                     # RANSAC

                     if (int(parameters['config']['ransac'])):
                         src_pts = np.float32([ kp_1[m.queryIdx].pt for m in good_points ]).reshape(-1,1,2)
                         dst_pts = np.float32([ kp_2[m.trainIdx].pt for m in good_points ]).reshape(-1,1,2)

                         # used to detect bad keypoints
                         M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                         matchesMask = mask.ravel().tolist()

                         h,w = primary_images[primary_count].image.shape[1], primary_images[primary_count].image.shape[0]
                         pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
                         dst = cv2.perspectiveTransform(pts,M)


                     # take smallest number of keypoints between two images
                     number_keypoints = 0
                     if len(kp_1) <= len(kp_2):
                         number_keypoints = len(kp_1)
                     else:
                         number_keypoints = len(kp_2)

                     # score boosting
                     score = score_boosting(primary_images[primary_count],
                        secondary_images[secondary_count], good_points, parameters)

                     # add the number of good points to score_matrix. start_i is
                     # passed in as a paramter to ensure that the correct row of the
                     # score matrix is being written to. Give this index the number
                     # of 'good_points' from the output of the KNN matcher.
                     score_matrix[start_i + primary_count][secondary_count] = score

                     # only do image processing if number of good points
                     # exceeeds threshold
                     if len(good_points) > write_threshold:
                         write_matches(kp_1, kp_2, good_points,
                            primary_images[primary_count], secondary_images[secondary_count],
                            image_destination)

                 except cv2.error as e:
                     print('\n\t\tERROR: {0}\n'.format(e))
                     print("\t\tError matching: " + os.path.basename(primary_images[primary_count].image_title) +
                         " and " + os.path.basename(secondary_images[secondary_count].image_title) + "\n")

    return score_matrix

################################################################################
def slice_generator(
        sequence_length,
        n_blocks):
    """ Creates a generator to get start/end indexes for dividing a
        sequence_length into n blocks
    """
    return ((int(round((b - 1) * sequence_length/n_blocks)),
             int(round(b * sequence_length/n_blocks)))
            for b in range(1, n_blocks+1))

################################################################################
def match_multi(primary_images, image_destination, n_threads, write_threshold, parameters):
    'Wrapper function for the "match". This also controls the multithreading'
    'if the user has declared to use multiple threads'

    # deep copy the primary_images for secondary images
    secondary_images = deepcopy(primary_images)

    # init score_matrix
    num_pictures = len(primary_images)
    score_matrix = np.zeros(shape = (num_pictures, num_pictures))

    # prep for multiprocessing; slices is a 2D array that specifies the
    # start and end array index for each program thread about to be created
    slices = slice_generator(num_pictures, n_threads)
    thread_list = list()

    print("\tImages to pattern match: {0}\n".format(str(num_pictures)))

    # start threading
    for i, (start_i, end_i) in enumerate(slices):

        # Ask Sultan about why primary_images turns into an np.ndarray instead of rec_list
        thread = threading.Thread(target = match,
                    args = (primary_images[start_i: end_i],
                            secondary_images,
                            image_destination,
                            start_i,
                            score_matrix,
                            write_threshold,
                            parameters))
        thread.start()
        thread_list.append(thread)

    for thread in thread_list:
        thread.join()

    return score_matrix
################################################################################
def add_cat_ID(rec_list, cluster_path):

    # create the list
    import pandas as pd
    csv_file = pd.read_csv(cluster_path)
    image_names = list(csv_file['Image Name'])
    cat_ID_list = list(csv_file['Cat ID'])

    for count in range(len(rec_list)):
        image = os.path.basename(rec_list[count].image_title)
        try:
            image_index = image_names.index(image)
        except ValueError:
            print('\tSomething is wrong with cluster_table file. Image name is not present.')

        cat_ID = cat_ID_list[image_index]
        rec_list[count].add_cat_ID(cat_ID)

    return rec_list

################################################################################
def crop(event, x, y, flags, param):

    global ref_points, cropping

    if event == cv2.EVENT_LBUTTONDOWN:
        ref_points = [(x, y)]
        cropping = True

    elif event == cv2.EVENT_LBUTTONUP:

        ref_points.append((x, y))
        cropping = False

        cv2.rectangle(param, ref_points[0], ref_points[1], (0, 255, 0), 2)
        cv2.imshow("image", param)

################################################################################
def manual_roi(rec_list, image_source):

    cropping = False
    count = 0

    #todo: do a check and make sure temp_templates folder isn't already created
    temp_templates = image_source.parents[1] / "temp_templates/"
    
    if (not os.path.exists(temp_templates)):
        os.mkdir(temp_templates)

    for i in glob.iglob(str(image_source)):
        print(i)
        # read in image and change size so it doesn't expand to whole screen; make a copy
        image = cv2.imread(i)
        image = cv2.resize(image, (960, 540))
        rec_list[count].add_image(i, image)
        image_clone = image.copy()

        # create an empy numpy array for the mask. Will be all black to start
        mask = np.zeros(image.shape, dtype = np.bool)

        # set up event callback for mouse click
        cv2.namedWindow("image")
        cv2.setMouseCallback("image", crop, image)

        while True:
            cv2.imshow("image", image)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("r"):
                image = image_clone.copy()

            if key == ord("c"):
                break

        if len(ref_points) == 2:

            # Using the coordinates from the moust click. Thurn an area of the mask to white.
            mask[ref_points[0][1]:ref_points[1][1], ref_points[0][0]:ref_points[1][0]] = True
            image = image_clone * (mask.astype(image_clone.dtype))
            locations = np.where(image != 0)
            image[locations[0], locations[1]] = (255, 255, 255)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            template_name = Path(i).with_suffix('.BMP')
            print(template_name.name)
            template_path = temp_templates / template_name.name
            print(template_path)
            cv2.imwrite(str(template_path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            rec_list[count].add_template(str(template_path), image)

        cv2.destroyAllWindows()
        count = count + 1

    return rec_list

################################################################################
def add_templates(rec_list, template_source):
    'Used for adding the premade templates to the recognition class if'
    'the user has them.'
    #MASK RCNN
    print(template_source, rec_list)
    config = snow_leopard.CustomConfig()
    BALLOON_DIR = os.path.join(ROOT_DIR, "C:/Users/Phil/SU-ECE-19-7-master-MaskRCNN/Recognition/samples/snow_leopard/dataset")
    class InferenceConfig(config.__class__):
    # Run detection on one image at a time
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

    config = InferenceConfig()
    config.display()
    # Device to load the neural network on.
    # Useful if you're training a model on the same 
    # machine, in which case use CPU and leave the
    # GPU for training.
    DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0
    # Inspect the model in training or inference modes
    # values: 'inference' or 'training'
    TEST_MODE = "inference"
    def get_ax(rows=1, cols=1, size=16):
        """Return a Matplotlib Axes array to be used in
        all visualizations in the notebook. Provide a
        central point to control graph sizes.
        
        Adjust the size attribute to control how big to render images
        """
        _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
        return ax
    # Load validation dataset
    dataset = snow_leopard.CustomDataset()
    dataset.load_custom(BALLOON_DIR, "val")

    # Must call before using the dataset
    dataset.prepare()

    print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))

    # Create model in inference mode
    with tf.device(DEVICE):
        model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
   
    # Set path to balloon weights file

    # Download file from the Releases page and set its path
    # https://github.com/matterport/Mask_RCNN/releases
    # weights_path = "/path/to/mask_rcnn_balloon.h5"

    # Or, load the last model you trained
    weights_path = os.path.join(ROOT_DIR, "C:/Users/Phil/SU-ECE-19-7-master-MaskRCNN/Recognition/Mask_RCNN-master/logs/bottle20200221T0110/mask_rcnn_bottle_0010.h5")

    # Load weights
    print("Loading weights ", weights_path)
    model.load_weights(weights_path, by_name=True)

  

    count = 0
    #todo: do a check and make sure temp_templates folder isn't already created
    temp_templates = template_source.parents[1] / "mrcnn_templates/"
    if (not os.path.exists(temp_templates)):
        os.mkdir(temp_templates)

    # add in template
    for t in glob.iglob(str(template_source)):
        

        IMAGE_DIR="C:/Users/Phil/SU-ECE-19-7-master-MaskRCNN/Image_Sets/quick_set/images"
        # Load a random image from the images folder
        file_names = next(os.walk(IMAGE_DIR))[2]
        maskimage = skimage.io.imread(rec_list[count].image_title)
        # Run detection
        results = model.detect([maskimage], verbose=1)

        # Visualize results
        r = results[0]
        ax = get_ax(1)
        bbb=visualize.display_instances(maskimage, r['rois'], r['masks'], r['class_ids'], 
                                    dataset.class_names, r['scores'], ax=ax,
                                    title="Predictions")
        display_images(np.transpose(r['masks'], [2, 0, 1]), cmap="binary")

 
        template = cv2.imread(t)
        r_mask = np.reshape(r['masks'], np.shape(r['masks'])[:2])
        r_mask = r_mask * 255
        print("r_mask shape: ", np.shape(r_mask))
        print(r_mask)

        #rec_list[count].add_template(t, r_mask)
        template_name = Path(t).with_suffix('.BMP')
        print(template_name.name)
        template_path = temp_templates / template_name.name
        print(template_path)
        cv2.imwrite(str(template_path), r_mask, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        rec_list[count].add_template(str(template_path), r_mask)

        print("dddddddddddd")
        print(rec_list[count].image_title)
        # This section is for the presentation only, remove later
        #temp1 = cv2.resize(rec_list[count].image, (960,540))
        #temp2 = cv2.resize(r_mask, (960,540))

        #horiz = np.hstack((temp1,temp2))
        #cv2.imshow("mathced image to tempalte", horiz)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        # end 

        count = count + 1

    return rec_list
################################################################################
def getTitleChars(title):
    'Used to pull the characteristics out of the image title name'
    title_chars = title.split("__")
    station = title_chars[1]
    camera = title_chars[2]
    date = title_chars[3]
    # dont want the last 7 characters
    time = title_chars[4][:-7]

    return station, camera, date, time
################################################################################
def init_Recognition(image_source, template_source):
    'Used to initalize a recongition object for each template/image pair'
    # # TODO: create a function that verifies the image and template names match

    rec_list = []
    count = 0

    # add images and templates in a parallel for-loop
    for i in glob.iglob(str(image_source)):

        # add new Recognition object to list
        rec_list.append(Recognition())

        # add image title and image to object
        image = cv2.imread(i)

        rec_list[count].add_image(i, image)

        # get title characteristics
        station, camera, date, time = getTitleChars(i)
        rec_list[count].add_title_chars(station, camera, date, time)

        # increment count
        count = count + 1

    # return the list of recognition objects
    return rec_list

########################### END FUNCTION DEFINITIONS ###########################

# MAIN
################################################################################
if __name__ == "__main__":

    # set up via command line
    parser = argparse.ArgumentParser()

    # easy config
    parser.add_argument("-work_directory", type = Path, required = False, default = None)

    # manual config
    parser.add_argument("-image_source", type = Path, required = True, default = None)
    parser.add_argument("-template_source", type = Path, required = False, default = None)
    parser.add_argument("-config_source", type = Path, required = False, default = None)
    parser.add_argument("-cluster_source", type = Path, required = False, default = None)
    parser.add_argument("-destination", type = Path, required = False, default = Path.cwd())
    parser.add_argument("-num_threads", type = int, required = False, default = 1)
    parser.add_argument("-write_threshold", type = int, required = False, default = 60)

    args = vars(parser.parse_args())

    # initialize depending on input arguments
    paths = {'images': '', 'templates': '', 'config': '', 'cluster': '', 'destination': ''}

    paths['images'] = args['image_source']
    paths['templates'] = args['template_source']
    paths['config'] = args['config_source']
    paths['cluster'] = args['cluster_source']
    paths['destination'] = args['destination']
    n_threads = args['num_threads']
    write_threshold = args['write_threshold']

    # TODO: change this to fit new command line argument scheme
    # Use the config.json file to import variable parameters
    with open(paths['config']) as config_file:
        parameters = json.load(config_file)

    # initialize the array of Recognition objects for the images
    rec_list = init_Recognition(paths['images'], paths['templates'])

    if (int(parameters['config']['templating']) == 1):
        print('\n\tUsing premade templates...\n')
        rec_list = add_templates(rec_list, paths['templates'])
    else:
        print('\n\tUsing the manual templating function...\n')
        rec_list = manual_roi(rec_list, paths['images'])

    # Get cat_ID information from cluster table
    if (paths['cluster'] != None):
        print("\n\tLoading information from cluster table...\n")
        # add in the cat ID data if available
        rec_list = add_cat_ID(rec_list, paths['cluster'])
    
    
    # START
    print("\tstarting matching process...\n")
    start = time.time()
    score_matrix = match_multi(rec_list, paths['destination'], n_threads, write_threshold, parameters)
    end = time.time()
    print("\tTime took to run: " + str((end - start)))

    # Normalize scores in matrix
    score_matrix = normalize_matrix(score_matrix)

    # write the score matrix to a .csv file
    print("\n\tWriting score_matrix to 'score_matrix.csv' to the destination folder...\n")
    np.savetxt(paths['destination'].joinpath('score_matrix.csv'), score_matrix, delimiter = ",")


    # check matrix for average hit/miss scores
    if (paths['cluster'] != None):
        print("Checking matrix...")
        check_matrix(rec_list, score_matrix)

    print('\n\tDone.\n')
    
############################### END  MAIN ######################################
