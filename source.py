



# import few important libraries
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt

# reading the image files

image = cv2.imread('C:/Users/DINESH/AppData/Local/Programs/PythonCodingPack/train/0000.tif')


# Convert the RBG images to GRAY
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Creating an empty data frame using pandas library
dataFrame = pd.DataFrame()

# to display one column 
image2 = image.reshape(-1)

# Adding feature  original pixel values to data frame as feature #1
dataFrame['Native Image'] = image2






# append Gabor features with Gabor filters
number = 1                                              # counting numbers for Gabor features in dataFrame
convolutionMatrix = []
for theta in range(3):                                  # Define numbers of thetas
    theta = theta/4. * np.pi 
    for sigma in (1, 3):                                # Sigma with 1 and 3
        for lamda in np.arange(0, np.pi, np.pi/4):      # Range of wavelenths
            for gamma in (0.05, 0.5):                   # Gamma values of 0.05 & 0.5 

                gabor_label = 'Gabor' + str(number)     # Label Gabor columns as Gabor1, ... print(gabor-label)

                ksize = 5
                kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_32F)

                convolutionMatrix.append(kernel)

                # add filterImage & values to new column
                filterImage = cv2.filter2D(image2, cv2.CV_8UC3, kernel)

                filteredImage = filterImage.reshape(-1)

                dataFrame['gabor_label'] = filteredImage  # Labels columns as Gabor1, Gabor2, ...

                print(gabor_label, ':theta=', theta, ':sigma=', sigma, ':lamda=', lamda, ':gamma=', gamma)

                number += 1                                # gradual increment for gabor column label

                

#################################################################################################################

# appending canny edge (edge detecting) with min and max value
border = cv2.Canny(image, 90, 180)
border1 = border.reshape(-1)
dataFrame['Canny Edge'] = border1

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


# adding ground truth vairables
labeledImage = cv2.imread('C:/Users/DINESH/AppData/Local/Programs/PythonCodingPack/train_mask/0000.tif')

# Convert the RBG images to GRAY
labeledImage = cv2.cvtColor(labeledImage, cv2.COLOR_BGR2GRAY)

# to display one column 
labeledImage2 = labeledImage.reshape(-1)

# Adding features like original pixel values to data frame as feature #1
dataFrame['Label'] = labeledImage2

print(dataFrame.head())


# define dependent variables
Y = dataFrame['Label'].values

# define independent variables
X = dataFrame.drop(labels=['Label'], axis=1)

#Split data into test and train
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.4, random_state=42)


# import ml algorithm and train the model
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=10, random_state=42)

# fitting the data
model.fit(X_train, Y_train)

# calculate the accuracy 
predictionTest = model.predict(X_test)


# import metrics as it assist to print or find out accuracy of training data

from sklearn import metrics

print("Accuracy=", metrics.accuracy_score(Y_test, predictionTest))

# to know which feature work best for random forest its built-in
# importances = list(model.feature_importances_)  # to print out the list of features
featuresList = list(X.columns)

feature_importance = pd.Series(model.feature_importances_, index=featuresList).sort_values(ascending=False)

print(feature_importance)

# how to save the model using pickle
import pickle 
fileName = 'random_forest_model'
pickle.dump(model, open(fileName, 'wb'))

# loading the saved model
loadModel = pickle.load(open(fileName, 'rb'))
outcome = loadModel.predict(X)
separated = outcome.reshape(image.shape)

# to visualize the prediction using built model
from matplotlib import pyplot as plt
plt.imshow(separated, cmap='jet')
plt.imsave('simulated_image.jpg', separated, cmap='jet')
