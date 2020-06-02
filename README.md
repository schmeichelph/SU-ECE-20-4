# SU-ECE-20-4

This repository holds the final deliverables of Seattle University Team ECE 20.4 for Panthera.

This will include a user guide for both the local version of Recognition.py as well as  

## Contents

The folder "Azure_Recognition_notebook" contains a user guide for the Azure Datbricks
version of Recognition.py. The Azure Databricks version is available in both ".py" and ".ipynb" if you are running
python notbook software.

The folder "Image_Sets" contains different sets of images that vary in size. They range from a set of 3 images to 
larger sets of 32 images. In these folders, a config.json file that relates to each individual photo set will be
there as well as its corresponding score_matrix.csv file. 

The folder "Recognition" contains both easy_run.py as well as Recognition.py. Also in this folder there is 
a user guide for how to set up and run Recognition.py

## Setup
Certain software needs to be installed in order to be able to run Recognition.py. Changes also need to be made
in Recognition.py to write to the correct folder, or depending on the operating system. 

### OpenCV:
  If you are running python 2, install OpenCV with 
  ```bash
  pip install opencv-contrib-python==3.3.1.11
  ```

  If you are running python 3, install OpenCV with 
  ```bash
  pip install opencv-contrib-python==3.4.2.16
  ```

### Tensorflow 
Tensorflow needs to be installed on your system as well and can be downloaded at this website: 
[Tensorflow](https://www.tensorflow.org/install)

### Markov Clustering
In order to install the correct software for markov clustering onto the system use the command:
  ```bash
  pip install markov_clustering"
  ```
### Line Changes
Specific paths must be changed depending on the operating system:
If running recognition on Windows, line 72 needs to be commented out, and line 74 needs to be uncommented.

## Usage

## TODOs

