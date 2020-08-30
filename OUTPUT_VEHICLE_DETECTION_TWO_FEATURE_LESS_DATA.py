


#PROJECT: CLASSIFY IMAGES AS VEHICLE/NOT VEHICLE USING MACHINE LEARNING MODELS.

#DATASET:2000 IMAGES( VEHICLE/NOT VEHICLE IMAGES)

#NUMBER OF FEATURES EXTRACTED:2

#MODEL USED: SVM

#Submitted by: KARISHMA UNNIKRISHNAN



#IMPORT LIBRARIES

import numpy as np
import cv2
from matplotlib import pyplot as plt
from skimage import feature

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics


#CHECK DATA

car_img=cv2.imread('car2.png')
car=cv2.cvtColor(car_img,cv2.COLOR_BGR2RGB)
plt.imshow(car)
plt.show()

not_car_img=cv2.imread('not_car2.png')
not_car=cv2.cvtColor(not_car_img,cv2.COLOR_BGR2RGB)
plt.imshow(not_car)
plt.show()


#EXTRACT FEATURES OF SINGLE IMAGE

hist_car=cv2.calcHist([car],[0,1,2],None,(32,32,32),[0,256,0,256,0,256])
hist_car.flatten()

hist_not_car=cv2.calcHist([not_car],[0,1,2],None,(32,32,32),[0,256,0,256,0,256])
hist_not_car.flatten()


img_car=cv2.resize(car,(32,32))
img_car_spat_bin=img_car.ravel()

img_not_car=cv2.resize(not_car,(32,32))
img_not_car_spat_bin=img_not_car.ravel()


out_car,hog_img_car=feature.hog(car,orientations=9,pixels_per_cell=(8,8),cells_per_block=(3,3),
                        visualize=True,transform_sqrt=True,feature_vector=True,multichannel=None)

out_not_car,hog_img_not_car=feature.hog(not_car,orientations=9,pixels_per_cell=(8,8),cells_per_block=(3,3),
                        visualize=True,transform_sqrt=True,feature_vector=True,multichannel=None)

#FUNCTION TO EXTRACT FETURES

#OUT OF THE ABOVE 3, HISTOGRAM OF GRADIENT WAS FOUND TO BETTER BECAUSE IT CLEARLY DEPICTS THE #EDGES AND ALSO THE DIRECTION IN WHICH IT CHANGES. THIS IS GOOD FOR IMAGES WITH LARGE FETURES.
#HERE TWO FEATURES WAS TAKEN AND DATASET WAS LIMITED TO 100, BECAUSE USING TWO FEATURES IS #COMPUTATIONALLY EXPENSIVE FOR MY SYSTEM(LAPTOP), SO HAD TO LIMIT THE DATASET.


def extract_features(image):
    out_feat=[]
    out,hog_img=feature.hog(image,orientations=9,pixels_per_cell=(8,8),cells_per_block=(3,3),
                        visualize=True,transform_sqrt=True,feature_vector=True,multichannel=None)
    out_feat.append(out)
    
    img=cv2.resize(image,(32,32))
    img_spat_bin=img.ravel()
    out_feat.append(img_spat_bin)
    
    #hist_img=cv2.calcHist([image],[0,1,2],None,(32,32,32),[0,256,0,256,0,256])
    #hist_img.flatten()
    #out_feat.append(hist_img)
    
    return np.concatenate(out_feat)


#EXTRACT IMAGES FROM FOLDER, FIND ITS FEATURES AND CREATE  FEATURE VECTORS.

import os
import glob

path_images=os.getcwd()

data=path_images+'/Cars_dataset2'
img_list=os.listdir(data)

features_car=[]
for img in img_list:
    input_img=cv2.imread(data+'/'+img)
    in_im=cv2.resize(input_img,(64,64))
    feature_hog=extract_features(in_im)
    features_car.append(feature_hog)

data1=path_images+'/Not_Car_dataset2'
img_list1=os.listdir(data1)

features_not_car=[]
for img1 in img_list1:
    input_img1=cv2.imread(data1+'/'+img1)
    in_im1=cv2.resize(input_img1,(64,64))
    feature_hog1=extract_features(in_im1)
    features_not_car.append(feature_hog1)

#TRAINING, FITTING AND TESTING MODEL
X=np.vstack((features_car,features_not_car))

y=np.hstack((np.ones(len(features_car)),np.zeros(len(features_not_car))))

X,y=shuffle(X,y,random_state=2)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=3)


model=SVC(kernel='linear',random_state=1)
model.fit(X_train,y_train)


y_pred=model.predict(X_test)


print(metrics.confusion_matrix(y_test,y_pred))

print(metrics.classification_report(y_test,y_pred))


#TESTING THE MODEL

input1_img=cv2.imread('car.png')
in_im1=cv2.resize(input1_img,(64,64))
feature_hog1=extract_features(in_im1)
test1=np.array(feature_hog1).reshape(1,-1)

in_im1=cv2.cvtColor(in_im1,cv2.COLOR_BGR2RGB)
plt.imshow(in_im1)

pred1=model.predict(test1)
if pred1==1:
    print('Its a car')
else:
    print('its not a car')


input_img2=cv2.imread('not_car.png')
in_im2=cv2.resize(input_img2,(64,64))
feature_hog2=extract_features(in_im2)
test2=np.array(feature_hog2).reshape(1,-1)

in_im2=cv2.cvtColor(in_im2,cv2.COLOR_BGR2RGB)
plt.imshow(in_im2)
plt.show()

pred2=model.predict(test2)
if pred2==1:
    print('Its a car')
else:
    print('its not a car')







