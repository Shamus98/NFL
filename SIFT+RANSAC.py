import numpy as np
import cv2 
from matplotlib import pyplot as plt
MIN_MATCH_COUNT = 10
img1 = cv2.imread('0.jpg',0)          # queryImage
img2 = cv2.imread('1.jpg',0) # trainImage
data = np.load("openpose.npy", allow_pickle = True)
kp = data.item()['57583_000082_Endzone.mp4']
X = []
Y = []
humans = []
humans_points = []
# Не получается удалить точки 
####################################################################
for i in range(len(kp)):    
    humans.append(kp[i].shape[0])
    humans_points.append(kp[i].shape[1])
for i in range(len(kp)):
    for j in range(humans[i]):
        if j != 0:
            x,y,w,h = int(min(X)), int(min(Y)), int(max(X)), int(max(Y))
            img1[y:h, x:w] = (255, 255, 255)
            img2[y:h,x:w] = (255, 255, 255)
        X = []
        Y = []
        for k in range(humans_points[i]):
            X.append(kp[i][j,k,0])
            Y.append(kp[i][j,k,1])
###############################################################################################
#img1 = cv2.rotate(img1, cv2.ROTATE_90_COUNTERCLOCKWISE) 
#img2 = cv2.rotate(img2, cv2.ROTATE_90_COUNTERCLOCKWISE)
# Initiate SIFT detector
sift = cv2.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1,des2,k=2)
# store all the good matches as per Lowe's ratio test.
good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)
if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()
    h,w = img1.shape[:2]
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)
    img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
else:
    print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
    matchesMask = None

draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)
img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
#vis = np.concatenate((img1, img2), axis=0)
#img3 = cv2.rotate(img3, cv2.ROTATE_90_CLOCKWISE)
#plt.imshow(img3, 'gray'),plt.show()

