import numpy as np
import cv2 
from matplotlib import pyplot as plt
MIN_MATCH_COUNT = 10
img1 = cv2.imread('0.jpg',0)          # queryImage
img2 = cv2.imread('1.jpg',0) # trainImage
data = np.load("openpose.npy", allow_pickle = True)
kp = data.item()['57583_000082_Endzone.mp4']
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
bad = []
proverka_bool = False
nj = kp[1].shape[0]
nk = kp[1].shape[1]
for m,n in matches:
    if m.distance < 0.7*n.distance:
        #print(kp2[m.trainIdx].pt)
        proverka = list(kp2[m.trainIdx].pt)
        bad.append(m)
        for j in range(nj):
            for k in range(nk):
                if (kp[1][j,k, 0] + 30 >= proverka[0]) and (kp[1][j,k, 0] - 30 <= proverka[0])and (kp[1][j,k, 1] + 30 >= proverka[1]) and (kp[1][j,k, 1] - 30 <= proverka[1]):
                    print('удаление координаты {0}, {1}'.format(proverka[0], proverka[1]))
                    proverka_bool = True
                    break
        if (proverka_bool != True):
            good.append(m)
        else:
            proverka_bool = False
if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    #ds = float(dst_pts[:,:,0][2551])
    #ks = float(dst_pts[:,:,1][2551])
    #print('{0} , {1}'.format(ds,ks))
    bad_dst_pts = np.float32([ kp2[m.trainIdx].pt for m in bad ]).reshape(-1,1,2)
    print(dst_pts[:,:,:].shape)
    # nj = kp[1].shape[0]
    # nk = kp[1].shape[1]
    # nsift = dst_pts[:,:,:].shape[0]
    #     for j in range(nj):
    #     for k in range(nk):
    #         for i in range(nsift):
    #             if (kp[1][j,k, 0] + 10 >= dst_pts[:,:,0][i]) and (kp[1][j,k, 0] - 10 <= dst_pts[:,:,0][i])and (kp[1][j,k, 1] + 10 >= dst_pts[:,:,1][i]) and (kp[1][j,k, 1] - 10 <= dst_pts[:,:,1][i]):
    #                 print('удаление координаты {0}'.format(dst_pts[:,:,:][i]))
    #             else:
    #                 np.append(dest_pts, dst_pts[:,:,:][i])
    # kp[i][j,k,0] is the X coordinate of the k-th keypoint of the j-th person (float in range 0..1280)
    # kp[i][j,k,1] is Y (float in range 0..720)
    # kp[i][j,k,2] is 1.0 if k-th keypoint is present, 0.0 otherwise
    # kp[i][j,k,3] is the confidence score in range 0..1
    #print(dest_pts.shape)
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
f, axarr = plt.subplots()
axarr.scatter(dst_pts[:,:,0], dst_pts[:,:,1],
           c = 'deeppink', s = 1)
axarr.imshow(img2),plt.show()
# f, badaxarr = plt.subplots() 
# badaxarr.scatter(bad_dst_pts[:,:,0], bad_dst_pts[:,:,1],
#            c = 'blue', s = 1)
# badaxarr.imshow(img2),plt.show()

