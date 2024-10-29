
'''
#Project: Image Segmentation
1. HOWTO: Ugrade Pip Install in Python Terminal
PS H:\CODING\PROJECT_IST\ImageSegmentation> python.exe -m pip install --upgrade pip      

'''

#USE TERMINAL: Powershell


#################### Define Libraries #####################################

#0. Install libraries in terminal:
#Write in terminal: pip install numpy
#Write in Terminal: pip install cv
#Write in Terminal: pip install opencv-python
#Write in terminal: matplotlib
#Import CSV library


#1. Install required Python packages libraries
import numpy as np #for linear algebra
import cv2
from matplotlib import pyplot as plt
import csv

    #%plt inline #Creates matplotlib commands inline to include within file


##################### DEFINE CONSTANTS #####################################









##################### GLOBAL VARIABLES ######################################

image = 0 #Image variable to hold image being segmented
image_pixel_dataset = [] #2D array of image variable







################### PSEUDOCODE #############################################

'''
1. Look for K number of segments in the image
2. Take an image to a feature space
3. Make K random initial means
4. Create K clusters by assigning for each point to the nearest initial mean
5. Re-compute the cluster’s means
6. If changes in all K means is less than a certain threshold e, stop. Else, re-compute clusters’ means until this condition is met. 

'''

############# ALGORITHM CLASS & FUNCTION DEFINITIONS ########################



def process_image(current_image): #1. Process Image
    #1. Import/insert image, prepare image in intended RGB colorscale
    #2. Prepare image into 2D Array

    ############# 1. PROCESS/DISPLAY IMAGE ###################
    #Change color to RGB (from BGR)
    #HOWTO: cv2.cvtColor(image VARIABLE, cv2.COLOR__MODE To MODE)
    current_image = cv2.cvtColor(current_image, cv2.COLOR_BGR2RGB)

    #Displays Image
    #HOWTO: cv2.imshow("browser intended title", image VARIABLE)
    cv2.imshow("window_image", current_image)

    #Waits for user to press any key
    #(Necessary step: to avoid Python kernel from crashing)
    cv2.waitKey(0)

    #Closing all open windows
    cv2.destroyAllWindows()

    ########### 2. CONVERT 3D IMAGE TO 2D shape ###################
    #Image is a 3D array (height, width, channels), so need to flatten
    #image into 2D array, where:
        #Rows = pixels
        #Columns = RGB values

    # To apply k-means clustering --> need to reshape to 2D array

    #Get dimensions of 3D image (height, width, RGBA format model/channel)
    print(image.shape)
    h, w, c = image.shape

    #Reshape image into 2D array of pixels and 2 color values (RGB)
    pixel_vals = current_image.reshape(h * w, c) #Result: 2D array with shape (height * width, 3)
    #where each row representes a pixel, and columns represent the RGB color channels (3)

    #Convert image's 2D array of pixels into float type
    #HOWTO: numpy's function np.float32(pixel_vals)
    pixel_vals = np.float32(pixel_vals)
    #print(pixel_vals) #COW
    #print(len(pixel_vals)) #COW Print Length of array

    return pixel_vals


def euclidean_distance(pixel1, pixel2):
    #calculates Euclidian distance between 2 pixels (EACH pixel point vs the MEAN)
    return np.sqrt(np.sum((pixel1 - pixel2)**2, axis = 1))




class KMeans():

    def __init__(self, n_clusters = 8, max_iter = 300):
        self.n_clusters = n_clusters #Initializing number of K clusters to SELF
        self.max_iter = max_iter #Initializing max number of iterations to SELF


    def initialize_centroids(self, data):
        #Randomly choose k data points (pixels) to be "initial centroids"
        #self.centroids = data[]


        return



    def assign_clusters_to_centroids(self, data):
        #Assign each data pixel point to their closest "initial centroid"




        return


    def update_centroid_means(self, data):
        #Calculate the center of the clusters
        #Calculate the distance of the data points from the center of each of the clusters
        #Depending on the distance of each data point of teh cluster, reassign the data 
        #points to teh nearest clusters

    
        return





    def evaluate_threshold():

        return






#################### MAIN CODE ##############################################


#### INSERT IMAGE #############
#Read in image
#HOWTO: cv2.imread('image file path')
image = cv2.imread('H:\\CODING\\PROJECT_IST\\flower.jpg')

#Process Image: PROCESS_IMAGE Function
image_pixel_dataset = process_image(image)

plt.imshow(image_pixel_dataset)
plt.xlabel("x")
plt.ylabel("y")

print(image_pixel_dataset)

































######################## OTHER NOTES ###############################

'''
#%matplotlib inline #Creates matplotlib commands inline to include within file


Explanation of #matplotlib inline
https://stackoverflow.com/questions/43027980/purpose-of-matplotlib-inline

Displaying images in browser code outline
+ Opening images in different color scales
https://www.geeksforgeeks.org/reading-image-opencv-using-python/

Introductory Code Assistance on K-Means Image sort and algorithm
https://www.geeksforgeeks.org/image-segmentation-using-k-means-clustering/



'''


