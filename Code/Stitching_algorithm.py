import cv2
import numpy as np
import math

img = cv2.imread('original_image.jpg')
#img = cv2.resize(img,(720,720))
#img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
bf = cv2.BFMatcher()
bf_1=cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
bf_2=cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)


def ImageEnhancement(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    imagenor = cv2.normalize(v, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    maxx = np.max(v)
    Mean, SD = cv2.meanStdDev(imagenor)
    D = 4 * SD
    gamma1 = -math.log(SD, 2)
    gamma2 = math.exp((1 - (Mean + SD)) / 2)
    if (D > 1 / 3):
        if (Mean < 0.5):
            imageout = pow(imagenor, gamma1) / (
                        pow(imagenor, gamma1) + ((1 - pow(imagenor, gamma1)) * pow(Mean, gamma1)))
        else:
            imageout = pow(imagenor, gamma1)
    else:
        k = pow(imagenor, gamma2) + (1 - pow(imagenor, gamma2)) * (pow(Mean, gamma2))
        c = 1 / 1 + (0.5 - Mean) * (k - 1)

        imageout = c * pow(imagenor, gamma2)

    imagedenor = np.multiply(imageout, maxx)
    imagedenor = np.uint8(imagedenor)
    merged = cv2.merge((h, s, imagedenor))

    enhancedimage = cv2.cvtColor(merged, cv2.COLOR_HSV2BGR)

    # Enhancing with CLAHE
    r, g, b = cv2.split(enhancedimage)

    channelr = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized1 = channelr.apply(r)

    channelg = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized2 = channelg.apply(g)

    channelb = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized3 = channelb.apply(b)

    EnhancedImage = cv2.merge((equalized1, equalized2, equalized3))
    return (EnhancedImage)


def CLAHE(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(3,3))
    return clahe.apply(image)


def AdaptiveGammaCorection(image):
    max = np.amax(image)
    image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    Avg, sigma = cv2.meanStdDev(image)
    D = 4 * sigma
    if D < (1 / 3):
        gamma = -(math.log2(sigma))
    else:
        gamma = math.exp((1 - (Avg + sigma)) / 2)

    GammaCorrected = np.power(image, gamma)

    if Avg >= 0.5:
        image = GammaCorrected
    else:
        image = GammaCorrected / (GammaCorrected + ((1 - GammaCorrected) * (np.power(Avg, gamma))))

    image = np.multiply(image, max - 2)
    image = image.astype("uint8")
    return image

#delta=((img_.shape[1]/3)*f)


def splicing(img_):
    img_spliced = []
    for i in range(0,N,1):
        name = "img_spliced_" + str(i)
        if i==0:
            img_spliced.append(img_[0:img_.shape[0],0:int(((img_.shape[1]) / N)+delta)])
            #  cv2.imshow(name, img_spliced[i])
        elif 0<i<N-1:
            img_spliced.append(img_[0:img_.shape[0],int((((img_.shape[1])/N)*i)-delta):int((((img_.shape[1])/N)*(i+1))+delta)])
            # cv2.imshow(name, img_spliced[i])
        else:
            img_spliced.append( img_[0:img_.shape[0], int((((img_.shape[1])/N)*i)-delta):img_.shape[1]])
            # cv2.imshow(name, img_spliced[i])
    img_spliced = img_spliced[::-1]

    yo = [ [ None for x in range(N) ] for y in range(M) ]
    print(yo)
    for j in range(0,N,1):
        img_ = img_spliced[j]
        for i in range(0,M,1):
            print(j,i)
            name = "img_spliced_" + str(i) + str(j)
            if i==0:
                yo[i][j]=img_[0:int(((img_.shape[0]) / M)+delta_2),0:img_.shape[1]]
                #cv2.imshow(name, yo[i][j])
            elif 0<i<M-1:
                yo[i][j]=img_[int((((img_.shape[0])/M)*i)-delta_2):int((((img_.shape[0])/M)*(i+1))+delta_2),0:img_.shape[1]]
                #cv2.imshow(name, yo[i][j] )
            else:
                yo[i][j]= img_[int((((img_.shape[0])/M)*i)-delta_2):img_.shape[0], 0: img_.shape[1] ]
            #cv2.imshow(name,  yo[i][j])

    return yo
def trim(frame):
    #crop top
    if not np.sum(frame[0]):
        return trim(frame[1:])
    #crop bottom
    elif not np.sum(frame[-1]):
        return trim(frame[:-2])
    #crop left
    elif not np.sum(frame[:,0]):
        return trim(frame[:,1:])
    #crop right
    elif not np.sum(frame[:,-1]):
        return trim(frame[:,:-2])
    return frame


#cv2.imshow("images",images)

def sift( img,img2 ):
    sift=cv2.xfeatures2d.SIFT_create()
    keypoints1,descriptor1=sift.detectAndCompute(img,None)
    keypoints_1,descriptor_1=sift.detectAndCompute(img2,None)
    img_keypoints1=cv2.drawKeypoints(img,keypoints1,None)
    img_keypoints_1=cv2.drawKeypoints(img2,keypoints_1,None)
    #sift_images = np.hstack((img_keypoints1, img_keypoints_1))
    #cv2.imshow("keypoints",sift_images)
    return [keypoints1 ,descriptor1,keypoints_1,descriptor_1]

def surf( img,img2 ):
    surf=cv2.xfeatures2d.SURF_create()
    keypoints2,descriptor2=surf.detectAndCompute(img,None)
    keypoints_2,descriptor_2=surf.detectAndCompute(img2,None)
    img_keypoints2=cv2.drawKeypoints(img,keypoints2,None)
    img_keypoints_2=cv2.drawKeypoints(img2,keypoints_2,None)
    #surf_images = np.hstack((img_keypoints2, img_keypoints_2))
    #cv2.imshow("keypoints",sift_images)
    return [keypoints2 ,descriptor2,keypoints_2,descriptor_2]

def orb( img,img2 ):
    orb = cv2.ORB_create()
    keypoints3,descriptor3=orb.detectAndCompute(img,None)
    keypoints_3,descriptor_3=orb.detectAndCompute(img2,None)
    img_keypoints3=cv2.drawKeypoints(img,keypoints3,None)
    img_keypoints_3=cv2.drawKeypoints(img2,keypoints_3,None)
    #orb_images = np.hstack((img_keypoints3, img_keypoints_3))
    #cv2.imshow("keypoints",sift_images)
    return [keypoints3 ,descriptor3,keypoints_3,descriptor_3]

def knn_bruteforce(bf,keypoints1 ,descriptor1,keypoints_1,descriptor_1,ratio_param ,img,img2):
    matches_3 = bf.knnMatch(descriptor1,descriptor_1,k=2)
    good = []
    # Apply ratio test (try different ratios)
    for ratio_param in np.arange(0.7,0.99,0.01):
        good = []
        for m,n in matches_3:
            if m.distance < ratio_param*n.distance:
                good.append(m)
        print(len(good))
        if len(good) >= MIN_MATCH_COUNT:
            break

    #bruteforce_knnmatcher

    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints_1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_param)
    h, w, t = img.shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    #dst = cv2.perspectiveTransform(pts, M)
    #img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
    #cv2.imshow("original_image_overlapping.jpg", img2)
    dst = cv2.warpPerspective(img,M,(img2.shape[1] + img.shape[1], img2.shape[0]))
    dst[0:img2.shape[0],0:img2.shape[1]] = img2
    return dst

#sift with bf

def bruteforce(bf_1,keypoints1 ,descriptor1,keypoints_1,descriptor_1,img,img2):
    matches_1=bf_1.match(descriptor1,descriptor_1)
    print(len(matches_1))
    matches_1 =sorted(matches_1, key=lambda x:x.distance)
    for m in matches_1:
        print(m.distance)
    #matching_result_1=cv2.drawMatches(img,keypoints1,img2,keypoints_1 ,matches_1, None,flags=2) #matches[:15]
    #cv2.imshow("Brute_force_matching ",matching_result_1)
    if len(matches_1) > MIN_MATCH_COUNT:
        pass
    else:
        print("Not enought matches are found - %d/%d", (len(matches_1) / MIN_MATCH_COUNT))
    src_pts_4 = np.float32([keypoints1[m.queryIdx].pt for m in matches_1]).reshape(-1, 1, 2)
    dst_pts_4 = np.float32([keypoints_1[m.trainIdx].pt for m in matches_1]).reshape(-1, 1, 2)
    M_4, mask_4 = cv2.findHomography(src_pts_4, dst_pts_4, cv2.RANSAC, ransac_param)
    dst_3 = cv2.warpPerspective(img,M_4,(img2.shape[1] + img.shape[1], img2.shape[0]))
    dst_3[0:img2.shape[0],0:img2.shape[1]] = img2
    return dst_3

#flann based
def flann_based(keypoints1 ,descriptor1,keypoints_1,descriptor_1,ratio_param,img,img2):
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50) #try check=100
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches_6 = flann.knnMatch(descriptor1,descriptor_1,k=2)
# Need to draw only good matches, so create a mask

    good_3=[]
# ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches_6):
        if m.distance < ratio_param*n.distance:
            good_3.append(m)
    print(len(good_3))
#flann= cv2.drawMatchesKnn(img,keypoints1,img2,keypoints_1,[good_3],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

#cv2.imshow('flann_based_sift',flann_sift)
    if len(good_3) > MIN_MATCH_COUNT:
        pass
    else:
        print("Not enought matches are found - %d/%d", (len(good_3) / MIN_MATCH_COUNT))
    src_pts_6 = np.float32([keypoints1[m.queryIdx].pt for m in good_3]).reshape(-1, 1, 2)
    dst_pts_6 = np.float32([keypoints_1[m.trainIdx].pt for m in good_3]).reshape(-1, 1, 2)
    M_6, mask_6 = cv2.findHomography(src_pts_6, dst_pts_6, cv2.RANSAC, ransac_param)
    dst_6= cv2.warpPerspective(img, M_6, (img2.shape[1] + img.shape[1], img2.shape[0]))
    dst_6[0:img2.shape[0], 0:img2.shape[1]] = img2
    return dst_6


def trim(frame):
    #crop top
    if not np.sum(frame[0]):
        return trim(frame[1:])
    #crop bottom
    elif not np.sum(frame[-1]):
        return trim(frame[:-2])
    #crop left
    elif not np.sum(frame[:,0]):
        return trim(frame[:,1:])
    #crop right
    elif not np.sum(frame[:,-1]):
        return trim(frame[:,:-2])
    return frame

N=3
f=0.35
ransac_param=9
ratio_param=0.75
MIN_MATCH_COUNT=1
img_ = ImageEnhancement(img)
h,w,t= img_.shape
M=3
f_2=0.35
delta_2= (h/M)*f_2

#delta=((img_.shape[1]/3)*f)
delta= (w/N)*f
yo=splicing(img_)
sift_op=[]
surf_op=[]
orb_op=[]
dst=[] #op of sift with knn_bf
M_rows=[None for x in range(M)]
yobo_yo=[]


# bf_knn with sift
for j in range(0,M,1):
    dst = []
    for i in range (1,N,1):
        if i==1:
            sift_op.append(sift(yo[j][0],yo[j][1]))
            dst.append ( knn_bruteforce(bf, sift_op[0][0], sift_op[0][1],sift_op[0][2], sift_op[0][3],ratio_param,yo[j][0],yo[j][1]))
            #cv2.imshow("Img" + str(j)+ str(0), yo[j][0])
            #cv2.imshow("Img" + str(j)+ str(1), yo[j][1])
            #cv2.waitKey(0)
        else:
           # cv2.imshow("Img" + str(j)+ str(i-2), dst[i-2])
            #cv2.imshow("Img" + str(j)+ str(i), yo[j][i])/
            sift_op.append(sift(dst[i-2],yo[j][i]))
            dst.append(knn_bruteforce(bf,sift_op[i-1][0],sift_op[i-1][1],sift_op[i-1][2],sift_op[i-1][3],ratio_param,dst[i-2],yo[j][i]))
    M_rows[j]=cv2.rotate(dst[len(dst)-1],cv2.ROTATE_90_CLOCKWISE)
    #cv2.imshow("Img" + str(j)+ str(0), yo[j][0])
    #cv2.imshow("row_no_"+str(j),M_rows[j])

sift_op = []
for i in range (1,M,1):
    if i==1:
        sift_op.append(sift(M_rows[0],M_rows[1]))
        yobo_yo.append ( knn_bruteforce(bf, sift_op[0][0], sift_op[0][1],sift_op[0][2], sift_op[0][3],ratio_param,M_rows[0],M_rows[1]))
    else:
        sift_op.append(sift(yobo_yo[i - 2], M_rows[i]))
        yobo_yo.append(knn_bruteforce(bf, sift_op[i - 1][0], sift_op[i - 1][1], sift_op[i - 1][2], sift_op[i - 1][3],ratio_param, yobo_yo[i - 2], M_rows[i]))
yobo_yo=cv2.rotate(trim(yobo_yo[len(yobo_yo)-1]),cv2.ROTATE_90_COUNTERCLOCKWISE)
cv2.imshow("final_stitched_image_sift_bf_knn",yobo_yo)
#cv2.imshow("Stitched_op_sift_bruteforce_knnmatcher", trim(dst[-1]))
cv2.imwrite("stitched_sift_bf_knn_matcher.jpg",yobo_yo)


#bf_knn with surf
M_rows=[None for x in range(M)]
yobo_yo=[]
for j in range(0,M,1):
    dst= []
    for i in range (1,N,1):
        if i==1:
            surf_op.append(surf(yo[j][0],yo[j][1]))
            dst.append ( knn_bruteforce(bf, surf_op[0][0], surf_op[0][1],surf_op[0][2], surf_op[0][3],ratio_param,yo[j][0],yo[j][1]))
        else:
            surf_op.append(surf(dst[i - 2], yo[j][i]))
            dst.append(knn_bruteforce(bf, surf_op[i - 1][0], surf_op[i - 1][1], surf_op[i - 1][2], surf_op[i - 1][3],
                                      ratio_param, dst[i - 2], yo[j][i]))
        M_rows[j] = cv2.rotate(dst[len(dst) - 1], cv2.ROTATE_90_CLOCKWISE)
surf_op = []
for i in range (1,M,1):
    if i==1:
        surf_op.append(surf(M_rows[0],M_rows[1]))
        yobo_yo.append ( knn_bruteforce(bf, surf_op[0][0], surf_op[0][1],surf_op[0][2], surf_op[0][3],ratio_param,M_rows[0],M_rows[1]))
    else:
        surf_op.append(surf(yobo_yo[i - 2], M_rows[i]))
        yobo_yo.append(knn_bruteforce(bf, surf_op[i - 1][0], surf_op[i - 1][1], surf_op[i - 1][2], surf_op[i - 1][3],ratio_param, yobo_yo[i - 2], M_rows[i]))
yobo_yo=cv2.rotate(trim(yobo_yo[len(yobo_yo)-1]),cv2.ROTATE_90_COUNTERCLOCKWISE)

cv2.imshow("final_stitched_image_surf_bf_knn",yobo_yo)
#cv2.imshow("Stitched_op_sift_bruteforce_knnmatcher", trim(dst[-1]))
cv2.imwrite("stitched_surf_bf_knn_matcher.jpg",yobo_yo)



#knn with orb
M_rows=[None for x in range(M)]
yobo_yo=[]

for j in range(0,M,1):
    dst = []
    for i in range (1,N,1):
        if i==1:
            orb_op.append(orb(yo[j][0],yo[j][1]))
            dst.append ( knn_bruteforce(bf, orb_op[0][0], orb_op[0][1],orb_op[0][2], orb_op[0][3],ratio_param,yo[j][0],yo[j][1]))

        else:
            orb_op.append(sift(dst[i-2],yo[j][i]))
            dst.append(knn_bruteforce(bf,orb_op[i-1][0],orb_op[i-1][1],orb_op[i-1][2],orb_op[i-1][3],ratio_param,dst[i-2],yo[j][i]))
    M_rows[j]=cv2.rotate(dst[len(dst)-1],cv2.ROTATE_90_CLOCKWISE)

orb_op = []
for i in range (1,M,1):
    if i==1:
        orb_op.append(orb(M_rows[0],M_rows[1]))
        yobo_yo.append ( knn_bruteforce(bf, orb_op[0][0], orb_op[0][1],orb_op[0][2], orb_op[0][3],ratio_param,M_rows[0],M_rows[1]))
    else:
        orb_op.append(orb(yobo_yo[i - 2], M_rows[i]))
        yobo_yo.append(knn_bruteforce(bf, orb_op[i - 1][0], orb_op[i - 1][1], orb_op[i - 1][2], orb_op[i - 1][3],ratio_param, yobo_yo[i - 2], M_rows[i]))
yobo_yo=cv2.rotate(trim(yobo_yo[len(yobo_yo)-1]),cv2.ROTATE_90_COUNTERCLOCKWISE)
cv2.imshow("final_stitched_image_orb_bf_knn",yobo_yo)
#cv2.imshow("Stitched_op_orb_bruteforce_knnmatcher", trim(dst[-1]))
cv2.imwrite("stitched_orb_bf_knn_matcher.jpg",yobo_yo)



#bf with sift
M_rows=[None for x in range(M)]
yobo_yo=[]
sift_op = []

for j in range(0,M,1):
    dst = []
    for i in range (1,N,1):
        if i==1:
            sift_op.append(sift(yo[j][0],yo[j][1]))
            dst.append (bruteforce(bf_1, sift_op[0][0], sift_op[0][1],sift_op[0][2], sift_op[0][3],yo[j][0],yo[j][1]))
        else:
            sift_op.append(sift(dst[i-2],yo[j][i]))
            dst.append(bruteforce(bf_1,sift_op[i-1][0],sift_op[i-1][1],sift_op[i-1][2],sift_op[i-1][3],dst[i-2],yo[j][i]))
    M_rows[j]=cv2.rotate(dst[len(dst)-1],cv2.ROTATE_90_CLOCKWISE)
sift_op = []
for i in range (1,M,1):
    if i==1:
        sift_op.append(sift(M_rows[0],M_rows[1]))
        yobo_yo.append ( bruteforce(bf_1, sift_op[0][0], sift_op[0][1],sift_op[0][2], sift_op[0][3],M_rows[0],M_rows[1]))
    else:
        sift_op.append(sift(yobo_yo[i - 2], M_rows[i]))
        yobo_yo.append(bruteforce(bf_1, sift_op[i - 1][0], sift_op[i - 1][1], sift_op[i - 1][2], sift_op[i - 1][3],yobo_yo[i - 2], M_rows[i]))
yobo_yo=cv2.rotate(trim(yobo_yo[len(yobo_yo)-1]),cv2.ROTATE_90_COUNTERCLOCKWISE)
cv2.imshow("final_stitched_image_sift_bf",yobo_yo)
#cv2.imshow("Stitched_op_sift_bruteforce_knnmatcher", trim(dst[-1]))
cv2.imwrite("stitched_sift_bf_matcher.jpg",yobo_yo)


#bf with surf
surf_op = []
M_rows=[None for x in range(M)]
yobo_yo=[]


for j in range(0,M,1):
    dst = []
    for i in range (1,N,1):
        if i==1:
            surf_op.append(surf(yo[j][0],yo[j][1]))
            dst.append (bruteforce(bf_1, surf_op[0][0], surf_op[0][1],surf_op[0][2], surf_op[0][3],yo[j][0],yo[j][1]))
        else:
            surf_op.append(surf(dst[i-2],yo[j][i]))
            dst.append(bruteforce(bf_1,surf_op[i-1][0],surf_op[i-1][1],surf_op[i-1][2],surf_op[i-1][3],dst[i-2],yo[j][i]))
    M_rows[j]=cv2.rotate(dst[len(dst)-1],cv2.ROTATE_90_CLOCKWISE)
surf_op = []
for i in range (1,M,1):
    if i==1:
        surf_op.append(surf(M_rows[0],M_rows[1]))
        yobo_yo.append ( bruteforce(bf_1, surf_op[0][0], surf_op[0][1],surf_op[0][2], surf_op[0][3],M_rows[0],M_rows[1]))
    else:
        surf_op.append(surf(yobo_yo[i - 2], M_rows[i]))
        yobo_yo.append(bruteforce(bf_1, surf_op[i - 1][0], surf_op[i - 1][1], surf_op[i - 1][2], surf_op[i - 1][3],yobo_yo[i - 2], M_rows[i]))
yobo_yo=cv2.rotate(trim(yobo_yo[len(yobo_yo)-1]),cv2.ROTATE_90_COUNTERCLOCKWISE)
cv2.imshow("final_stitched_image_surf_bf",yobo_yo)
#cv2.imshow("Stitched_op_sift_bruteforce_knnmatcher", trim(dst[-1]))
cv2.imwrite("stitched_surf_bf_matcher.jpg",yobo_yo)



#bf with orb
orb_op = []
M_rows=[None for x in range(M)]
yobo_yo=[]

for j in range(0,M,1):
    dst = []
    for i in range (1,N,1):
        if i==1:
            orb_op.append(orb(yo[j][0],yo[j][1]))
            dst.append (bruteforce(bf_2, sift_op[0][0], orb_op[0][1],orb_op[0][2], orb_op[0][3],yo[j][0],yo[j][1]))

        else:
            orb_op.append(orb(dst[i-2],yo[j][i]))
            dst.append(bruteforce(bf_2,orb_op[i-1][0],orb_op[i-1][1],orb_op[i-1][2],orb_op[i-1][3],dst[i-2],yo[j][i]))
    M_rows[j]=cv2.rotate(dst[len(dst)-1],cv2.ROTATE_90_CLOCKWISE)

orb_op = []
for i in range (1,M,1):
    if i==1:
        orb_op.append(orb(M_rows[0],M_rows[1]))
        yobo_yo.append ( bruteforce(bf_2, orb_op[0][0], orb_op[0][1],orb_op[0][2], orb_op[0][3],M_rows[0],M_rows[1]))
    else:
        orb_op.append(orb(yobo_yo[i - 2], M_rows[i]))
        yobo_yo.append(bruteforce(bf_2, orb_op[i - 1][0], orb_op[i - 1][1], orb_op[i - 1][2], orb_op[i - 1][3],yobo_yo[i - 2], M_rows[i]))
yobo_yo=cv2.rotate(trim(yobo_yo[len(yobo_yo)-1]),cv2.ROTATE_90_COUNTERCLOCKWISE)
cv2.imshow("final_stitched_image_orb_bf",yobo_yo)
#cv2.imshow("Stitched_op_sift_bruteforce_knnmatcher", trim(dst[-1]))
cv2.imwrite("stitched_orb_bf_matcher.jpg",yobo_yo)



#flann_based with sift
sift_op = []
M_rows=[None for x in range(M)]
yobo_yo=[]


for j in range(0,M,1):
    dst = []
    for i in range (1,N,1):
        if i==1:
            sift_op.append(sift(yo[j][0],yo[j][1]))
            dst.append ( flann_based( sift_op[0][0], sift_op[0][1],sift_op[0][2], sift_op[0][3],ratio_param,yo[j][0],yo[j][1]))
        else:
            sift_op.append(sift(dst[i-2],yo[j][i]))
            dst.append(flann_based(sift_op[i-1][0],sift_op[i-1][1],sift_op[i-1][2],sift_op[i-1][3],ratio_param,dst[i-2],yo[j][i]))
    M_rows[j]=cv2.rotate(dst[len(dst)-1],cv2.ROTATE_90_CLOCKWISE)

sift_op = []
for i in range (1,M,1):
    if i==1:
        sift_op.append(sift(M_rows[0],M_rows[1]))
        yobo_yo.append ( flann_based( sift_op[0][0], sift_op[0][1],sift_op[0][2], sift_op[0][3],ratio_param,M_rows[0],M_rows[1]))
    else:
        sift_op.append(sift(yobo_yo[i - 2], M_rows[i]))
        yobo_yo.append(flann_based( sift_op[i - 1][0], sift_op[i - 1][1], sift_op[i - 1][2], sift_op[i - 1][3],ratio_param, yobo_yo[i - 2], M_rows[i]))
yobo_yo=cv2.rotate(trim(yobo_yo[len(yobo_yo)-1]),cv2.ROTATE_90_COUNTERCLOCKWISE)
cv2.imshow("final_stitched_image_sift_flann_based_knn",yobo_yo)
#cv2.imshow("Stitched_op_sift_bruteforce_knnmatcher", trim(dst[-1]))
cv2.imwrite("stitched_sift_flann_basedknn_matcher.jpg",yobo_yo)



#flann_based with surf
M_rows=[None for x in range(M)]
yobo_yo=[]
surf_op = []


for j in range(0,M,1):
    dst= []
    for i in range (1,N,1):
        if i==1:
            surf_op.append(surf(yo[j][0],yo[j][1]))
            dst.append ( flann_based( surf_op[0][0], surf_op[0][1],surf_op[0][2], surf_op[0][3],ratio_param,yo[j][0],yo[j][1]))
        else:
            surf_op.append(surf(dst[i - 2], yo[j][i]))
            dst.append(flann_based( surf_op[i - 1][0], surf_op[i - 1][1], surf_op[i - 1][2], surf_op[i - 1][3],
                                      ratio_param, dst[i - 2], yo[j][i]))
        M_rows[j] = cv2.rotate(dst[len(dst) - 1], cv2.ROTATE_90_CLOCKWISE)
surf_op = []
for i in range (1,M,1):
    if i==1:
        surf_op.append(surf(M_rows[0],M_rows[1]))
        yobo_yo.append ( flann_based( surf_op[0][0], surf_op[0][1],surf_op[0][2], surf_op[0][3],ratio_param,M_rows[0],M_rows[1]))
    else:
        surf_op.append(surf(yobo_yo[i - 2], M_rows[i]))
        yobo_yo.append(flann_based( surf_op[i - 1][0], surf_op[i - 1][1], surf_op[i - 1][2], surf_op[i - 1][3],ratio_param, yobo_yo[i - 2], M_rows[i]))
yobo_yo=cv2.rotate(trim(yobo_yo[len(yobo_yo)-1]),cv2.ROTATE_90_COUNTERCLOCKWISE)

cv2.imshow("final_stitched_image_surf_flann_based_knn",yobo_yo)
#cv2.imshow("Stitched_op_sift_bruteforce_knnmatcher", trim(dst[-1]))
cv2.imwrite("stitched_surf_flann_based_knn_matcher.jpg",yobo_yo)

cv2.waitKey(0)
cv2.destroyAllWindows()
