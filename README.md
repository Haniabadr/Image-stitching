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
-------------------------------------------------------------------------------------------------------------------------
# Tuning Parameters:
**We used some tuning parameters that we need to insert in the code first before runing it in order to be able to control the stitching process** <br>
1- N (number of horizontal images) <br>
2- M (number of vertical images) <br>
3- F (percentage of overlapping area from the spliced image) <br>
4- Delta (horizontally overlapping area) , delta=(W/N)*F <br>
5- Delta 2 (vertically overlapping area),   delta2=(H/M)*F <br>
6- Minimum number of matches <br>
7- RANSAC threshold value <br>
8- Ratio for LOWES Ratio test <br>
-The upcoming results depends on these parameters, so if we changed them the results may change. <br>

# Results:
* The following image shows the input image, then the enhanced input image then the result after slicing it into nine images then the following image show the result of the stitching process.

![results](https://user-images.githubusercontent.com/103740764/170628042-655d5fd2-ac46-4973-895d-acb82bd0905c.PNG)
-------------------------------------------------------------------------------------------------------------------------------


![results_2](https://user-images.githubusercontent.com/103740764/170628874-a1442626-c8e8-417e-a3a3-9d921550fbb5.PNG)

To see the results of the nine combinations check our presentation 
https://docs.google.com/presentation/d/1H4gMagRwKGyAdIOS6C-PiwEj1j8JAgDd/edit?usp=sharing&ouid=101559812285838275889&rtpof=true&sd=true

# Conclusion:
* From the previous results we conclude that SIFT with KNN Flann based matcher is the best combination as it has the fastest feature matching and the best accuracy, also its noted that the ORB with its three combinations gave the worst results . <br>
Important note that this results was under certain tunning parameters which means that if these parameters changed these results may change accordingly , the overlapping area were set to 25% of the original image and there original image was spliced to nine images (M=N=3) .
Our algorithm proved that it can work accurately with min of f=0.08
Trade off between speed and increasing number of images to be stitched.





