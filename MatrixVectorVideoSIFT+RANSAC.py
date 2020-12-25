import numpy as np
import cv2 
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D

MIN_MATCH_COUNT = 10
data = np.load("openpose.npy", allow_pickle = True)
kp = data.item()['57583_000082_Endzone.mp4']
cap = cv2.VideoCapture('/Users/vanurin/Desktop/Kaggle/NFL/57583_000082_Endzone.mp4')
count = 0
prev_frame = cv2.imread('0.jpg',0)
#prev_frame = cv2.rotate(prev_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)  
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        frame90 = frame
        #frame90 = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        sift = cv2.SIFT_create()
        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(frame90,None)
        kp2, des2 = sift.detectAndCompute(prev_frame,None)
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1,des2,k=2)
        matchesMask = [[0,0] for i in range(len(matches))]
        # store all the good matches as per Lowe's ratio test.
        good = []
        for i,(m,n) in enumerate(matches):
            if m.distance < 0.7*n.distance:
                matchesMask[i]=[1,0]
                good.append(m)
        dst_pt = [ kp2[m.trainIdx].pt for m in good ]
        if len(good)>MIN_MATCH_COUNT:
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
            matchesMask = mask.ravel().tolist()
            h,w = frame90.shape[:2]
            pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
            dst = cv2.perspectiveTransform(pts,M)
            prev_frame = cv2.polylines(prev_frame,[np.int32(dst)],True,255,3, cv2.LINE_AA)
        else:
            print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
            matchesMask = None

        draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                        singlePointColor = None,
                        matchesMask = matchesMask, # draw only inliers
                        flags = 2)
        frame_output = cv2.drawMatches(frame90,kp1,prev_frame,kp2,good,None,**draw_params)
        #frame_output = cv2.rotate(frame_output, cv2.ROTATE_90_CLOCKWISE)
        plt.imshow(frame_output, 'gray'),plt.show()
        #mpimg.imsave('/Users/vanurin/Desktop/Kaggle/NFL/Ransacvideo_frame_57583_000082_Endzone/{:d}.jpg'.format(count), frame_output)
        K1 = [[1,1,1],[0,1,1],[0,0,1]]
        K = np.float32(K1)
        num, Rs, Ts, Ns  = cv2.decomposeHomographyMat(M, K)
        for i, Rt in enumerate(zip(Rs, Ts)):
            R, t = Rt
            print("option " + str(i+1))
            rvec, _ = cv2.Rodrigues(R)
            print('rvec=')
            print(rvec)
            if i==3:
                if count==0:
                    matrixrotates = np.array(R)*np.array(t)
                else:
                    matrixrotates = matrixrotates*np.array(R)*np.array(t)
            print('ts=')
            print(t)
        print('  ################### {0} ################## '.format(count))
        print(matrixrotates)
        count += 1 # i.e. at 30 fps, this advances one second
        cap.set(1, count)
        prev_frame = frame90.copy()
    else:
        cap.release()
        break
    