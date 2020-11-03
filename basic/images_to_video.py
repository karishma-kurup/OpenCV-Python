import cv2

import os
from os.path import isfile, join

pathIn= 'C:/Users/user/Documents/Computer Vision/Cat'
image_folder = pathIn 
video_name = 'myvideo.avi'
os.chdir(pathIn) 

images = [img for img in os.listdir(image_folder) 
          if img.endswith(".jpg") or
             img.endswith(".jpeg") or
             img.endswith("png")] 


print(images)  

frame = cv2.imread(os.path.join(image_folder, images[0])) 


height, width, layers = frame.shape   

video = cv2.VideoWriter(video_name, 0, 1, (width, height))  


for image in images:  
    video.write(cv2.imread(os.path.join(image_folder, image)))  


cv2.destroyAllWindows()  
video.release()  
  

