# Image-stitching

# Objective:
This project is part of a bigger project which is capturing the retina and classifying these images in to healthy, diabetic or glaucomatic to capture these images we used a 20D lens which has a small field of view and based on the surveys that we have done with several opthalmologists they needed a panoromic view of the retina in order to be able to diagnose it so we decided to take several pictures of different spots of the retina with the help of the eye tracking algorithm  and then stitch them back together, so this was the motivation behind building our algorithm,this algorithm stitches any number of images vertically, horizontally and both together.

# Algorithm architecture:

* We applied image enhancement techniques like Clahe and adaptive gamma correction **(this was a research paper that we converted to a code)** and it made the code alot   more generic and enhanced it's performance alot
* then we used Feature extraction methods to extract the features of the image we tried three different methods **(SIFT ,SURF and ORB)**
* We used Feature matching methods after this to match these extracted Features, we tried three different methods **(brute force matcher, brute force KNN and Flann based using KNN)** 
*  
