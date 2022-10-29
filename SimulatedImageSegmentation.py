# @ authoer: Dinesh Kumar Gupta


# import few important libraries
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt

# reading the image files

image = cv2.imread('C:/Users/DINESH/AppData/Local/Programs/PythonCodingPack/train/0000.tif')


# Convert the RBG images to GRAY
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def feature_extractions(image):

    # Creating an empty data frame using pandas library
    dataFrame = pd.DataFrame()

    # to display one column 
    image2 = image.reshape(-1)

    # Adding features like original pixel values to data frame as feature #1
    dataFrame['Native Image'] = image2

    # print(dataFrame.head())


    # appending first set- Gabor filters and its features
    number = 1                                                              # counting numbers for Gabor features in dataFrame
    convolutionMatrix = []
    for theta in range(2):                                                  # Define numbers of thetas
        theta = theta/4. * np.pi 
        for sigma in (1, 3):                                                # Sigma with 1 and 3
            for lamda in np.arange(0, np.pi, np.pi/4):                      # Range of wavelenths
                for gamma in (0.05, 0.5):                                   # Gamma values of 0.05 & 0.5 

                    gabor_label = 'Gabor' + str(number)                     # Label Gabor columns as Gabor1, ... print(gabor-label)

                    ksize = 5
                    kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_32F)

                    convolutionMatrix.append(kernel)

                    # add filterImage and add values to new column
                    filterImage = cv2.filter2D(image2, cv2.CV_8UC3, kernel)

                    filteredImage = filterImage.reshape(-1)

                    dataFrame['gabor_label'] = filteredImage                 # Labels columns as Gabor1, Gabor2, ...

                    print(gabor_label, ':theta=', theta, ':sigma=', sigma, ':lamda=', lamda, ':gamma=', gamma)

                    number += 1                                               # gradual increment for gabor column label

                    #print(dataFrame.head())

#################################################################################################################

    # appending canny edge (edge detecting) with min and max value
    borders = cv2.Canny(image, 100, 200)
    borders1 = borders.reshape(-1)
    dataFrame['Canny Edge'] = borders1

    print(dataFrame.head())

    from scipy import ndimage as nd

    #creating Gaussian with sigma = 3
    gaussianImage = nd.gaussian_filter(image, sigma=3)
    gaussianImage1 = gaussianImage.reshape(-1)
    dataFrame['Gaussian Sigma3'] = gaussianImage1

    # creating Gaussian with sigma = 7
    gaussianImage2 = nd.gaussian_filter(image, sigma=7)
    gaussianImage3 = gaussianImage2.reshape(-1)
    dataFrame['Gaussian Sigma7'] = gaussianImage3

    # creating Median with size = 3
    medianImage = nd.median_filter(image, size=3)
    medianImage1 = medianImage.reshape(-1)
    dataFrame['Median Sigma3'] = medianImage1

    # creating Variance with size = 3
    #variance_image = nd.generic_filter(image, np.var, size=3)
    #variance_image1 = variance_image.reshape(-1)
    #dataFrame['Variance Sigma3'] = variance_image1

    #print(dataFrame.head())

    return dataFrame
    
##################################################################################

import glob
import pickle
from matplotlib import pyplot as plt

filename = 'simulatedImage_model'

load_model = pickle.load(open(filename, 'rb'))

path = 'image/Program/Python/'
for file in glob.glob(path):    # go through each file
    image1 = cv2.imread(file)   # reading the image
    image = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)   # converting the image to gray

    X = feature_extractions(image)
    result = load_model.predict(X)                      # load the saved model using pickle
    segmented = result.reshape((image.shape))
    imageName = file.split("e_")   # split the imageName
    plt.imsave('images/Segmented/' + imageName[1], segmented, cmap='jet')   # save the segmented images

    

