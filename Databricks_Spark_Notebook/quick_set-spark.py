
# MAGIC %md
# MAGIC ### Mount the working directory container (Azure-specific)
# MAGIC from Azure Databricks documentation 
# MAGIC https://docs.databricks.com/data/data-sources/azure/azure-datalake-gen2.html#mount-azure-data-lake-storage-gen2-filesystem

# COMMAND ----------

%scala
// We created an Enterprise Application called 'RecognitionDevs'. 
// This was used to get an application-id and to generate a secret as well as access the storage account.
// [Azure Active Directory > Manage > Enterprise Applications]
val configs = Map(
  "fs.azure.account.auth.type" -> "OAuth",
  "fs.azure.account.oauth.provider.type" -> "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider",
  "fs.azure.account.oauth2.client.id" -> "6e252b88-8ce2-4e3b-a45a-234259aa54b0",
  "fs.azure.account.oauth2.client.secret" -> "s0:kK7K6DZIQva-Rj@VeXN7bg.1=u.b3",
  "fs.azure.account.oauth2.client.endpoint" -> "https://login.microsoftonline.com/6fae808c-42ee-4074-bd88-b51c2011953a/oauth2/token")

// It was important to give this Enterprise Application the role 'Storage Blob Data Contributor' to our storage 
// account so we did not get a 403 Forbidden error when opening or reading files.
// [Storage Account > Access control (IAM) > Role Assignments > Add > Role > Storage Blob Data Contributor > Select 'RecognitionDevs']
dbutils.fs.mount(
  source = "abfss://quickset-workingdirectory@recognitionsa.dfs.core.windows.net/",
  //source = "abfss://tenimages-workingdirectory@recognitionsa.dfs.core.windows.net/",
  mountPoint = "/mnt/quick_set",
  //mountPoint = "/mnt/ten_images",
  extraConfigs = configs)


# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ## Recognition program ported into Spark

# COMMAND ----------


"""
Authors: Ross Pitman (Panthera)
         Jack Gularte, Devin DeWitt (ECE 19.7), 
         Sultan Alneif, Anthony Caballero (ECE 20.4),
         Philip Schmeichel, Amudhan Sekar (ECE 20.4 on Azure Databricks)
File: quick_set-spark.ipynb
Date Last Modified: May 23rd, 2020
For: Panthera Organization
Purpose: Use computer vision to help determine the number of individual cats in
            a database.
Output: 'score_matrix'; A folder containing a matrix of similarity scores between
            each two images within the database
To Run: Import notebook into Databricks and run with cluster configured as in NOTES.
            Check the mounted storage account for the score matrix CSV file and 
            download for use with clustering algorithm.
"""

# NOTES
################################################################################
"""
       * Databricks cluster configuration during development:
           - Databricks Runtime version 6.4 (Scala 2.11, Spark 2.4.5)
           - Disabled autoscaling
           - Terminate after 20 minutes of inactivity
           - Worker Type: Standard_DS3_v2, 4 workers
           - Driver Type: Standard_DS5_v2
           
       * Libraries:
           - numpy
           - opencv-contrib-python-headless==3.4.2.17
           - scikit-image
           - IPython[all]
           
       * Storage account: ADLS StorageV2 (general purpose v2)
           - Hierarchical namespace enabled under [Advanced > Data Lake Storage Gen2]
"""
##### END NOTES ################################################################

##### IMPORTS ##################################################################
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
import matplotlib.pyplot as plt
##### END IMPORTS ##############################################################

##### RECOGNITION CLASS DEFINITION #############################################
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

##### FUNCTION DEFINITIONS #####################################################
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
            # don't check the same image.
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
    max_matrix = score_matrix.max()

    # normalize
    score_matrix = score_matrix

    # add identity matrix
    score_matrix = score_matrix + np.identity(len(score_matrix[1]))
    return score_matrix

################################################################################
def write_matches(kp_1, kp_2, good_points, primary_image, secondary_image, image_destination):
    'This function takes the output of the KNN matches and draws all the matching points'
    'between the two images. Writes the final product to the output directory'

    # TODO: Call this function to write the output image into destination folder with drawings of keypoints and matches 
    
    # parameters to pass into drawing function
    draw_params = dict(matchColor = (0,255,0),
                       singlePointColor = (255,0,0),
                       flags = 0)

    # draw the matches between two upper pictures and horizontally concatenate
    result = cv2.drawMatches(
        primary_image.image,
        kp_1,
        secondary_image.image,
        kp_2,
        good_points,
        None,
        **draw_params) # draw connections

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
    image_path = image_destination + (str(len(good_points)) +
    "___" +
    re.sub(".jpg", "", os.path.basename(primary_image.image_title)) +
    "___" +
    re.sub(".jpg", ".JPG", os.path.basename(secondary_image.image_title))
    )

    # Finally, write the finished image to the output folder.
    cv2.imwrite(str(image_path), result_vertical_concat, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
################################################################################
def score_boosting(primary_image, secondary_image, good_points):
    'uses image characteristics to boost scores'
    score = len(good_points)

    if (primary_image.station == secondary_image.station):
        if (primary_image.camera == secondary_image.camera):
            if (primary_image.date == secondary_image.date):
                score = score * float(2.5)
            else:
                score = score * float(2.0)
        else:
            score = score * float(1.5)

    return score
################################################################################
def match(primary_images, secondary_images, image_destination,
            start_i, score_matrix, write_threshold, parameters):
    'main function used for determining matches between two images.'
    'Finds the sift keypoints/descriptors and uses a KNN based matcher'
    'to filter out bad keypoints. Writes final output to score_matrix'
    # Begin loop on the primary images to match. Due to multithreading of the
    # program, this may not be the full set of images.
    for primary_count in range(len(primary_images)):
        
        #print("\t\tMatching: " + os.path.basename(primary_images[primary_count].image_title) + "\n")

        # create mask from template and place over image to reduce ROI
        mask_1 = cv2.imread(primary_images[primary_count].template_title, -1) 
        mySift = cv2.xfeatures2d.SIFT_create()
        kp_1, desc_1 = mySift.detectAndCompute(primary_images[primary_count].image, mask_1)

        # parameter setup and create nearest-neighbor matcher
        index_params = dict(algorithm = 0, trees = 5)
        search_params = dict()
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        # Begin nested loop for the images to be matched to. This secondary loop
        # will always iterate over the full dataset of images.
        for secondary_count in range(len(secondary_images)):
            print("matching the image " + str(primary_count)+ " with image " +str(secondary_count))
            # check if same image; if not, go into sophisticated matching
            if primary_images[primary_count].image_title != secondary_images[secondary_count].image_title:  

                 # create mask from template
                 mask_2 = cv2.imread(secondary_images[secondary_count].template_title, -1)   
                 #cv2.imshow("image",rec_list[secondary_count].image)
                 time.sleep(10)
                 kp_2, desc_2 = mySift.detectAndCompute(secondary_images[secondary_count].image, mask_2)
                
                 #print("Secondary image", secondary_image)
                 #cv2.imshow(secondary_image)
                 
                 # check for matches
                 try:
                     # Check for similarities between pairs
                     matches = flann.knnMatch(desc_1, desc_2, k=2)

                     # Use Lowe's ratio test
                     good_points = []
                     for m, n in matches:
                         if m.distance < 0.7 * n.distance:
                             good_points.append(m)


                     # take smallest number of keypoints between two images
                     number_keypoints = 0
                     if len(kp_1) <= len(kp_2):
                         number_keypoints = len(kp_1)
                     else:
                         number_keypoints = len(kp_2)

                     # score boosting
                     score = score_boosting(primary_images[primary_count],
                        secondary_images[secondary_count], good_points)

                     # add the number of good points to score_matrix. start_i is
                     # passed in as a parameter to ensure that the correct row of the
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
                     print("\t\tError matching: " + primary_images[primary_count].image_title +
                         " and " + secondary_images[secondary_count].image_title + "\n")   

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

    ## TODO: SWITCH TO MULTIPROCESSING LIBRARY IN PYTHON TO UTILIZE THE MULTIPLE WORKERS IN A DATABRICKS CLUSTER
    ##       Ross Pitman's original version of Recognition was implemented with multiprocessing. This article
    ##       from Medium gives a lead on using Python multiprocessing libraries with Databricks:
    ##       https://towardsdatascience.com/3-methods-for-parallelization-in-spark-6a1a4333b473
    # start threading
    for i, (start_i, end_i) in enumerate(slices):

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
        print("appending thread "+ str(i))
    for thread in thread_list:
        thread.join()
        print("joining thread")

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
        
################################################################################
def variance_of_laplacian(image):
	# compute the Laplacian of the image and then return the focus
	# measure, which is simply the variance of the Laplacian
	return cv2.Laplacian(image, cv2.CV_64F).var()

################################################################################
def add_templates(rec_list, template_source):
    'Used for adding the premade templates to the recognition class if'
    'the user has them.'
    TEMPLATE_FOLDER = r"/dbfs/mnt/ten_images/templates/"
    mount_folder = r"dbfs:" + template_source
    template_source = r"/dbfs" + template_source
    count = 0
    
    # add in template
    for t in dbutils.fs.ls(str(mount_folder)):
        print("template", count, ": ", t)
        
        # add image title and image to object
        template = cv2.imread(template_source + t.name)
        rec_list[count].add_template(t.name, template)

        count = count + 1

    return rec_list
################################################################################
def getTitleChars(title):
    'Used to pull the characteristics out of the image title name'
    title_chars = title.split("__")
    station = title_chars[1]
    camera = title_chars[2]
    date = title_chars[3]
    # don't want the last 7 characters
    time = title_chars[4][:-7]

    return station, camera, date, time

################################################################################
def init_Recognition(image_source, template_source):

    IMAGE_FOLDER = r"/dbfs/mnt/quick_set/images/"
    mount_folder = r"dbfs:" + image_source
    image_source = r"/dbfs" + image_source
    rec_list = []
    count = 0

    # add images and templates in a parallel for-loop 
    for i in dbutils.fs.ls(mount_folder):
        print("image", count, ": ", i)
        
        # add new Recognition object to list
        rec_list.append(Recognition())

        # add image title and image to object
        image = cv2.imread(image_source + i.name)
        rec_list[count].add_image(i.name, image)

        # get title characteristics
        station, camera, date, time = getTitleChars(i.name)
        rec_list[count].add_title_chars(station, camera, date, time)

        # increment count
        count = count + 1

    # return the list of recognition objects
    return rec_list


# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ## cv2.imshow() stand-in (unused)
# MAGIC function for displaying images in a Spark notebook
# MAGIC From Jonathan Scholtes' Stochastic Coder blog:
# MAGIC https://stochasticcoder.com/2018/06/06/python-image-processing-on-azure-databricks-part-1-opencv-image-compare/ 

# COMMAND ----------

def plot_img(figtitle,subtitle,img1,img2,site):
  
  #create figure with std size
  fig = plt.figure(figtitle, figsize=(10, 5))
  
  plt.suptitle(subtitle,fontsize=24)
  
  ax = fig.add_subplot(1, 2, 1)  
  # base is hardcoded for img1
  ax.set_title("Base",fontsize=12)
  plt.imshow(img1)
  plt.axis("off")
  
  ax = fig.add_subplot(1, 2, 2)
  # site is used in site iteration
  ax.set_title(site,fontsize=12)
  plt.imshow(img2)
  plt.axis("off")

  display(plt.show())

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ## START OF RECOGNITION MAIN

# COMMAND ----------

# Set up configs with filepaths based on the mount
paths = {'images': '', 'templates': '', 'config': '', 'cluster': '', 'destination': ''}
paths['images'] = "/mnt/quick_set/images/"
paths['templates'] = "/mnt/quick_set/templates"
paths['config'] = "dbfs:/mnt/quick_set/config.json"
paths['cluster'] = None
paths['destination'] = "dbfs:/mnt/destination"
n_threads = 1
write_threshold = 30
parameters = spark.read.json('dbfs:/mnt/quick_set/config.json', multiLine = True)

# COMMAND ----------

# create recognition objects, given images and templates
rec_list = init_Recognition(paths['images'], paths['templates'])
rec_list = add_templates(rec_list, paths['templates'])

# COMMAND ----------

# keypoint matching to generate score_matrix
score_matrix = match_multi(rec_list, paths['destination'], n_threads, write_threshold, parameters)
print(score_matrix)

# COMMAND ----------

# write score_matrix as CSV (should work for both AWS and Azure)
zrdd = spark.sparkContext.parallelize(score_matrix)
df = zrdd.map(lambda x: x.tolist()).toDF([])
df.coalesce(1).write.format("com.databricks.spark.csv").option("header", "false").mode ("overwrite").save("mnt/quick_set/score_matrix")
display(dbutils.fs.ls("dbfs:/mnt/quick_set/score_matrix/"))

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ## END OF RECOGNITION MAIN

# COMMAND ----------

# clean up cluster by removing mount point
dbutils.fs.unmount("/mnt/quick_set/")
#dbutils.fs.unmount("/mnt/ten_images/")

# COMMAND ----------
