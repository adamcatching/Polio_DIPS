"""
Benjamin Adam Catching
Andino Lab, Summer Rotation
Polio-virus DIPs Project
2018-07-23
"""

# This script will contain useful functions for processing mice IR videos

# Import packages
import numpy as np
from PIL import Image, ImageDraw
import cv2
import skvideo.io
import skimage.filters
import skimage.measure


def video_correction(checkerboard_vertices,
                     image_test=False):

    """
    Images are calibrated from a matrix of vertices that are mapped to
    a checkerboard that is used to calculate the values required for
    image correction. These values are output as K, D, and DIM.

    Parameters
    ----------
    checkerboard_vertices:
        A 2-D array of (x, y) points, from which a checkerboard can be
        constructed

        e.g. [[(x11, y11), (x12, y12), ... (x1M, y1M)],
                                       ...
                                       ...
              [(xN1, yN1), (xN2, yN2), ... (xNM, yNM)]]

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

    # Create a checkerboard out of the vertices given

    # Create a blank background for the image (gray-scale)
    seg_image = Image.new('RGB', (1920, 1080), (125, 125, 125))

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
            ImageDraw.Draw(seg_image).polygon(temp_poly, fill=color)

    # Convert the background image to numpy array
    gray = np.array(seg_image)
    print(gray.shape)
    # Test that the mask is working
    if image_test:
        return gray

    # Use the background image to create K, D, and DIM

    # Try all values of the checkerboard
    for i in range(8, 5, -1):
        for j in range(12, 7, -1):
            checkerboard = (6, 8)

            subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                               30, 0.1)

            objp = np.zeros((1, checkerboard[0] * checkerboard[1], 3), np.float32)
            objp[0, :, :2] = np.mgrid[0:checkerboard[0], 0:checkerboard[1]].T.reshape(-1, 2)

            objpoints = []  # 3d point in real world space
            imgpoints = []  # 2d points in image plane.

            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray,
                                                     checkerboard,
                                                     cv2.CALIB_CB_ADAPTIVE_THRESH +
                                                     cv2.CALIB_CB_FAST_CHECK +
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
        A 2-D, gray-scale numpy array

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
                                                       (h, w),
                                                       cv2.CV_16SC2)

    # Undistort the image
    undistorted_image = cv2.remap(image,
                                  map_x,
                                  map_y,
                                  interpolation=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT)

    return undistorted_image


def seg_mice(ir_image):
    """
    From the 2-D, gray-scale numpy array, segment the mice-like
    objects. This is done using a local threshold

    Parameters
    ----------
    ir_image:
        A 2-D 8-bit numpy array

    Return
    ------
    seg_image:
        A 2-D binary image of the mice
    """

    # Define the size of the local threshold
    block_size = 1001

    # Find the local theshold value for each block in the image
    local_thresh = skimage.filters.threshold_local(ir_image,
                                                   block_size,
                                                   offset=10)
    # Threshold image
    binary_local = ir_image < local_thresh
    # Label the segmented regions
    image_labeled, number_labels = skimage.measure.label(binary_local, background=0, return_num=True)
    # Get the properties of the labeled regions
    image_props = skimage.measure.regionprops(image_labeled)

    # Create a blank region of the original image
    seg_image = np.zeros((len(binary_local), len(binary_local[0])))
    # Go through props
    for index, prop in enumerate(image_props):
        # print(prop.area)
        # If the region properties are within the threshold
        if prop.area >= 20000 and prop.eccentricity <= 0.95:
            # Select the region
            # print(index)
            temp_seg = image_labeled == index + 1
            filled_seg = temp_seg
            # Add the temp region
            seg_image = seg_image + filled_seg

    return seg_image


class MouseVideoProcess:
    """
    Mouse video object, read in as a video file path, process it for
    later analysis.
    """

    def __init__(self, video_path, num_frames, distort_param=0):
        """
        From the location of the video (.m4v format required), return
        an array of undistorted images, called images. These images
        can then be used for further processing

        Parameters
        ----------
        video_path:
            Path to the video to be processed. Must be in .m4v format

        num_frames:
            The number of frames of the image to use.

        distort_param:
            If of the form [K, D], undistort the images using the
            function 'undistort'. If the list is empty, it is assumed
            that the video file is undistorted and the array of images
            is saved.

        Return
        ------
        images:
            A 3-D array where the first dimension is the time dimension
            and the other two are of the image [t, x, y].
        """

        self.video_path = video_path
        self.raw_video = skvideo.io.vreader(video_path)
        self.frames = []
        # Assign the set number of frames to self.frames
        for i in range(num_frames):
            temp_frame = next(self.raw_video)[:, :, 0]
            if distort_param != 0:
                temp_frame = undistort(temp_frame,
                                       distort_param[0],
                                       distort_param[1])
            self.frames.append(temp_frame)
        self.frames = np.array(list(x) for x in self.frames)

    def rough_segment(self):
        """
        Using the undistorted frames, perform segmentation on each on
        to find a rough idea of where the mice are and are not.

        Return
        ------
        segmented_frames
            3-D, binary numpy array. The first index is time, the
            second index is x, and the third index is y
        """

        # Initialize the list that will hold the frames
        segmented_frames = []
        # For each frame, segment each frame
        for frame in self.frames:
            temp_segmented_frame = seg_mice(frame)
            segmented_frames.append(temp_segmented_frame)

        # Redefine the list of numpy arrays as a 3-D numpy array
        segmented_frames = np.array(list(x) for x in segmented_frames)

        return segmented_frames
