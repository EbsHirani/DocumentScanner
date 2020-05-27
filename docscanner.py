import numpy as np 
import cv2

PATH = ""
HEIGHT = 800
WIDTH = 450

def nothing(x): #funtion passed to trackbars
    pass

cv2.namedWindow("Trackbars")
cv2.resizeWindow("Trackbars", 450, 300)
cv2.createTrackbar("t1", "Trackbars", 200, 255, nothing)
cv2.createTrackbar("t1", "Trackbars", 200, 255, nothing)

while True : 
	blank = np.zeros((HEIGHT, WIDTH, 3), dtype = np.uint8)
	img = cv2.resize(c2.imread(PATH), WIDTH, HEIGHT)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	blur = cv2.GaussianBlur(gray, (5,5), 1)

	t1 = cv2.getTrackbarPos("t1", "Trackbars")
    t2 = cv2.getTrackbarPos("t2", "Trackbars")

    thresh = cv2.Canny(blur, t1, t2)

    dil = cv2.dilate(thresh, np.ones((5,5)), iterations = 2)
    er = cv2.erode(dil, np.ones((5,5)), iterations = 1)

    contours, hierarchy = cv2.findContours(er, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 

	max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 5000:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area

    try :
    	biggest = biggest.reshape((4, 2))
	    rec = np.zeros((4, 1, 2), dtype=np.int32)
	    add = biggest.sum(1)
	 
	    rec[0] = biggest[np.argmin(add)]
	    rec[3] =biggest[np.argmax(add)]
	    diff = np.diff(biggest, axis=1)
	    rec[1] =biggest[np.argmin(diff)]
	    rec[2] = biggest[np.argmax(diff)]
        pts1 = np.float32(rec) 
        pts2 = np.float32([[0, 0],[WIDTH, 0], [0, HEIGHT],[WIDTH, HEIGHT]]) 
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        warp = cv2.warpPerspective(gray, matrix, (WIDTH, HEIGHT))
 
        warp=warp[20:warp.shape[0] - 20, 20:warp.shape[1] - 20]
        warp = cv2.resize(warp,(WIDTH,HEIGHT))
 	
        adapted= cv2.adaptiveThreshold(warp, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        adapted = cv2.bitwise_not(adapted)
        adapted=cv2.medianBlur(adapted,3)

        cv2.imshow('Result', adapted)

        if cv2.waitKey(1) &amp; 0xFF == ord('s'):
        	cv2.imwrite("saved.jpg",adapted)
        	break

cv2.destroyAllWindows()
