"""
Benjamin Adam Catching
Andino Lab, Summer Rotation
Poliovirus DIPs Project
2018-07-23
"""

# This script will contain useful functions for processing mice IR videos

# Import packages
import numpy as np
from PIL import Image, ImageDraw
import cv2

def video_correction(checkerboard_vertices,
                     image_shape=(1920, 1080),
                     image_test = False):

    """
    Images are calibrated from a matrix of vertices that are mapped to
    a checkerboard that is used to calculate the values required for
    image correction. These values are output as K, D, and DIM.

    Parameters
    ----------
    checkerboard_vertices":
        A 2-D array of (x, y) points, from which a checkerboard can be
        constructed

        e.g. [[(x11, y11), (x12, y12), ... (x1M, y1M)],
                                       ...
                                       ...
              [(xN1, yN1), (xN2, yN2), ... (xNM, yNM)]]

    image_shape:
        Unless defined, the shape of the images in the video is
        1920x1080

    image_test:
        If true, return the numpy grayscale image of the checkerboard,
        otherwise return the normal values

    Return
    ------
    K:
        A 2-D numpy array with the the diagonal elements representing
        the x, y and skew values of the focal lengths and the optical
        centers in the first two rows of the third column.
    D:
        A 2-D matrix of size 1x4, where the first two values are
        linear distortion coefficients and the last two values are
        correction factors the camera not being perfectly parallel to
        the surface.
    DIM:
        A length-2 tuple containing the dimensions of the image. This
        can be acquired by image.shape if the image is a numpy array.
    """

    # Check to make sure a numpy array is input
    try:
        type(checkerboard_vertices) == type(np.array([]))
    except ValueError:
        print('Use a numpy array for matrix of vertices')

    # Create a checkerboard out of the vertices given

    # Create a blank background for the image (grayscale)
    blank_background = Image.new('RGB', (1920, 1080), (125, 125, 125))

    # Iterate over the vertices, creating an alternating black-white square
    n = len(checkerboard_vertices)
    m = len(checkerboard_vertices[0])
    for i in range(n-1):
        for j in range(m-1):
            # Define the color to use
            if (i+j) % 2 == 0:
                color = (255, 255, 255)
            else:
                color = (0, 0, 0)
            # Define the vertices of the current polygon
            np_temp_poly = [
                checkerboard_vertices[i, j],
                checkerboard_vertices[i, j+1],
                checkerboard_vertices[i+1, j+1],
                checkerboard_vertices[i+1, j]
            ]
            # Convert the array of numpy arrays to list of tuples
            temp_poly = tuple(map(tuple, np_temp_poly))
            # Draw the polygon
            ImageDraw.Draw(blank_background).polygon(temp_poly, fill=color)

    # Convert the background image to numpy array
    gray = np.array(blank_background)
    print(gray.shape)
    # Test that the mask is working
    if image_test:
        return gray

    # Use the background image to create K, D, and DIM

    # Try all values of the checkerboard
    for i in range(8, 5, -1):
        for j in range(12, 7, -1):
            CHECKERBOARD = (6, 8)

            subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                               30, 0.1)
            calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + \
                                cv2.fisheye.CALIB_CHECK_COND + \
                                cv2.fisheye.CALIB_FIX_SKEW

            objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
            objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

            objpoints = []  # 3d point in real world space
            imgpoints = []  # 2d points in image plane.

            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray,
                                                     CHECKERBOARD,
                                                     cv2.CALIB_CB_ADAPTIVE_THRESH + \
                                                     cv2.CALIB_CB_FAST_CHECK + \
                                                     cv2.CALIB_CB_NORMALIZE_IMAGE)

            if ret:
                print(i, j)
                # If found, add object points, image points (after refining them)
                objpoints.append(objp)
                cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), subpix_criteria)
                imgpoints.append(corners)
                # Try
                ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints,
                                                                   imgpoints,
                                                                   gray.shape[::-1],
                                                                   None,
                                                                   None)
                """ 
                N_OK = len(objpoints)
                K = np.zeros((3, 3))
                D = np.zeros((4, 1))
                rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
                tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
                rms, _, _, _, _ = \
                    cv2.fisheye.calibrate(
                        objpoints,
                        imgpoints,
                        gray.shape[::-1],
                        K,
                        D,
                        rvecs,
                        tvecs,
                        calibration_flags,
                        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
                    )
                """
                break
            if ret:
                break
    if not image_test and ret:
        return mtx, dist, gray.shape[:2]
    elif not image_test:
        print('Checkerboard not found')


def undistort(image, K, D):
    """
    Undistort the input image using the K and D parameters input.

    Parameters
    ----------
    image:
        A 2-D, grayscale numpy array

    K:
        A 2-D numpy array with the the diagonal elements representing
        the x, y and skew values of the focal lengths and the optical
        centers in the first two rows of the third column.

    D:
        A 2-D matrix of size 1x4, where the first two values are
        linear distortion coefficients and the last two values are
        correction factors the camera not being perfectly parallel to
        the surface.

    Return
    ------
    undistorted_image:
        A 2-D, grayscale numpy array with distortion removed.
    """

    # Extract the height (h) and width (w) dimensions of the image
    h, w = image.shape[:2]

    # Create the two map parameters to deconvolute the input image
    map_x, map_y = cv2.fisheye.initUndistortRectifyMap(K,
                                                     D,
                                                     np.eye(3),
                                                     K,
                                                     DIM,
                                                     cv2.CV_16SC2)

    # Undistort the image
    undistorted_img = cv2.remap(image,
                                map_x,
                                map_y,
                                interpolation=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT)

    return undistorted_image

class MouseVideo:
    """Mouse video object, read in as a """

    def __init__(self, video_link, distort_param=[]):
        """
        From the location of the video (.m4v format required)
        """