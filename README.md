# Image-stitching

# Objective:
This project is part of a bigger project which is capturing the retina and classifying these images in to healthy, diabetic or glaucomatic to capture these images we used a 20D lens which has a small field of view and based on the surveys that we have done with several opthalmologists they needed a panoromic view of the retina in order to be able to diagnose it so we decided to take several pictures of different spots of the retina with the help of the eye tracking algorithm  and then stitch them back together, so this was the motivation behind building our algorithm,this algorithm stitches any number of images vertically, horizontally and both together.

# Algorithm architecture:

* We applied image enhancement techniques like Clahe and adaptive gamma correction **(this was a research paper that we converted to a code)** and it made the code alot   more generic and enhanced it's performance alot
* then we used Feature extraction methods to extract the features of the image we tried three different methods **(SIFT ,SURF and ORB)**
* We used Feature matching methods after this to match these extracted Features, we tried three different methods **(brute force matcher, brute force KNN and Flann based using KNN)** 
* Mismatches were removed then using **RANSAC** 
* then we applied **Homography** 
* after this **image warping** was applied 
* and last step was trimming the black areas resulting from the stitching process around the retina 

This algorithm can stitch any number of images ,  so for example if you're stitching three images it will stitch first two together then the resulting image will be stitched with the third image

**as a result of trying three Feature extraction methods with three Feature matching methods we got nine different combinations/algorithms**

# Simulation:
As i mentioned earlier the objective of this code is stitching multiple retinal images to get a panoramic view and as the mechanism that captures the retinal images wasn't ready yet when we developed this code and we wanted to test it **so we developed a piece of code that slices the image into any desired number of images** so that we can stitch them back together and see if it's working properly, **in order to be able to stitch any two images together there must be an overlaping area or common area between them so that the algorithm would find common features between them to match, so we controlled this area while slicing the images using some parameters**

**the illustration below shows what we mean by overlapping area between images incase we are stitching nine images together which is just an example to illustrate the idea behind the code but you can stitch any number of images you want, the N images is the number of images horizontally, the M images is the number of imgages vertically, delta is the overlapping area between two images horizontaly and delta 2 is overlaping area between two images vertically**



![Untitled Diagram (2)](https://user-images.githubusercontent.com/103740764/170622806-1b39af81-5d23-4c18-a396-b0b9ba6741a7.png)

# For more illustrations and details about the algorithm :

## Feature Extraction:


![Capture1_1](https://user-images.githubusercontent.com/103740764/170625065-add03ce9-77a1-428b-bdc9-95afb988050c.PNG)

## Feature Matching:
![Capture1_2](https://user-images.githubusercontent.com/103740764/170625193-a1261d5a-8a4b-499e-9e11-0d0cefc98b9c.PNG)
-------------------------------------------------------------------------------------------------------------------------


![Capture_2 (3)](https://user-images.githubusercontent.com/103740764/170625592-49edb8f7-6a38-4828-afda-7da387d5a4e9.PNG)
-------------------------------------------------------------------------------------------------------------------------


![Capture_3 (2)](https://user-images.githubusercontent.com/103740764/170625768-d5fae187-37d2-40e3-bd41-b0fbd493a1a2.PNG)
-------------------------------------------------------------------------------------------------------------------------


![Capture_4 (2)](https://user-images.githubusercontent.com/103740764/170625825-92d6495a-998b-45e1-b103-9654c02ee1fe.PNG)
-------------------------------------------------------------------------------------------------------------------------


![Capture_5 (2)](https://user-images.githubusercontent.com/103740764/170625895-4a8db20b-8581-40cf-a574-4cc3e6b5660d.PNG)
