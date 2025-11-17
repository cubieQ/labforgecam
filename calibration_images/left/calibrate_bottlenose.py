import numpy as np
import cv2 as cv
import glob

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points for 9x6 chessboard (CORRECTED)
objp = np.zeros((9*6,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('*.bmp')

print(f"Found {len(images)} images")
print("Starting calibrating loop for 10 images")

found_count = 0

for fname in images:
    img = cv.imread(fname)
    if img is None:
        print(f"Could not read image: {fname}")
        continue
        
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    print("Processing image:", fname)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (9,6), None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        found_count += 1
        print("Found corners in image:", fname)
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        cv.drawChessboardCorners(img, (9,6), corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(500)
    else:
        print(f"No corners found in: {fname}")

cv.destroyAllWindows()

print(f"Successfully processed {found_count} out of {len(images)} images")