"""
Benjamin Adam Catching
Andino Lab, Summer Rotation
Polio DIPs Project
2018/07/17
"""

# The ultimate goal of this script is to make it usable on on any computer
# through terminal. Multi-well analysis will be added once test images are
# acquired using cell cytosol marker (CY5) and either dead cell marker (SYTOX)
# or Polio-virus marker (GFP).

# Import necessary packages based on the mini-conda environment
# For data display
import matplotlib.pyplot as plt
import seaborn as sns

# Necessary for analysis
import numpy as np
import skimage.measure
import skimage.filters
import skimage.morphology
import skimage.io
import skimage.segmentation
import skimage.exposure
import skimage.feature
import scipy.ndimage


class BulkDroplet:
    """Create an image object from the read in file"""

    def __init__(self, filename, micron_per_pixel=1, gfp_thresh=0,
                 multi_channel=False):
        """Initialize the image data"""

        # Define input image attributes
        self.filename = filename
        self.image = skimage.io.imread(filename)
        self.micron_per_pixel = micron_per_pixel
        self.gfp_thresh = gfp_thresh
        self.shape = self.image.shape
        self.multi_channel = multi_channel

    def droplet_segment(self, testing=False, bright_channel=0):
        """Return droplets and their properties"""

        print('The file is updated')
        # If the image has multiple channels, choose the channel to determine droplets from
        if self.multi_channel:
            image_bright = self.image[:, :, bright_channel]
        else:
            image_bright = self.image

        # Find the Otsu threshold
        bright_thresh_otsu = skimage.filters.threshold_otsu(image_bright)

        # Label thesholded images
        bright_threshold = image_bright > bright_thresh_otsu
        image_labeled, number_labels = skimage.measure.label(bright_threshold, background=0, return_num=True)

        # Get the properties of the labeled regions
        image_props = skimage.measure.regionprops(image_labeled)

        # Create a blank region of the original image
        blank_background = np.zeros(image_bright.shape)

        for index, prop in enumerate(image_props):
            # print(prop.area)
            # If the region properties are within the threshold
            if prop.area >= 10000 and prop.eccentricity <= 0.4:
                # Select the region
                # print(index)
                """temp_seg = image_labeled==index+1
                filled_seg = temp_seg"""
                # Set the center of the circle
                (center_x, center_y) = prop.centroid
                radius = prop.major_axis_length / 2
                for (x, y), value in np.ndenumerate(blank_background):
                    dist_x = center_x - x
                    dist_y = center_y - y
                    dist = np.sqrt(dist_x ** 2 + dist_y ** 2)
                    if dist < radius:
                        blank_background[x, y] = 1
                # Add the temp region
                # blank_background = blank_background + filled_seg

        # Fill the holes of the image
        image_droplets = scipy.ndimage.binary_fill_holes(blank_background)
        image_droplets = skimage.morphology.closing(image_droplets, selem=skimage.morphology.disk(2))
        image_droplets = skimage.morphology.erosion(image_droplets, selem=skimage.morphology.disk(2))

        # If testing is True, show the image
        if testing:
            with sns.axes_style("dark"):
                # Define the number of pixels in a millimeter
                mm = (1 / self.micron_per_pixel) * 500

                # Define the number of ticks
                num_x_ticks = image_bright.shape[1] // mm + 1
                num_y_ticks = image_bright.shape[0] // mm + 1
                plt.imshow(image_droplets)
                plt.xticks(np.arange(0, image_bright.shape[1], mm), np.arange(0, num_x_ticks))
                plt.yticks(np.arange(0, image_bright.shape[0], mm), np.arange(0, num_y_ticks))
                plt.xlabel("x-axis (mm)")
                plt.ylabel("y-axis (mm)")
                plt.title("Segmented Droplets")
                plt.show()

        image_labeled, number_labels = skimage.measure.label(image_droplets, background=0, return_num=True)

        # Get the properties of the labeled regions
        image_props = skimage.measure.regionprops(image_labeled, image_bright)

        # Display the number of complete droplets
        print()

        return image_labeled, image_props


def cells_from_droplet(labeled_image, raw_bright, droplet_num):
    """
    From segmented black-white droplets, the brightfield image; single channel, and the
    selection of which droplet, return a black-white mask of the cells in the droplet
    """

    bright_droplet_props = skimage.measure.regionprops(labeled_image, raw_bright)

    # List of segmented droplets
    bright_droplets = []
    droplet_masks = []

    for index, prop in enumerate(bright_droplet_props):
        bright_droplets.append(prop.intensity_image)
        droplet_masks.append(prop.image)

    # Assign images
    cell_droplet = bright_droplets[droplet_num]

    # Remove any hot pixels
    selem = skimage.morphology.disk(3)
    cell_droplet_median = skimage.filters.median(cell_droplet, selem)

    # Create a gaussian blur of the image and subtract from median image
    cell_droplet_gaussian_blur = skimage.filters.gaussian(cell_droplet_median, sigma=15)
    cell_droplet_sub_gaussian = cell_droplet_median - cell_droplet_gaussian_blur

    # Perform a Scharr operation on the no cell droplet
    cell_droplet_temp_scharr = skimage.filters.scharr(cell_droplet_sub_gaussian, droplet_masks[droplet_num])
    # Otsu threshold the scharr image
    cell_droplet_thresh = skimage.filters.threshold_otsu(cell_droplet_temp_scharr)
    # Fill holes created from the otsu threshold
    cell_droplet_filled = scipy.ndimage.binary_fill_holes(cell_droplet_temp_scharr > cell_droplet_thresh)
    # Try to fill any partial no_cell_filled
    blur_droplet_cells = skimage.filters.gaussian(cell_droplet_filled, 2)
    smooth_droplet_cells = blur_droplet_cells > .25

    # Now that objects have been thresholded in the droplets, label and get props
    cell_droplet_labels = skimage.measure.label(smooth_droplet_cells, background=0, return_num=False)

    # Get regionprops and filter based on them
    cell_droplet_props = skimage.measure.regionprops(cell_droplet_labels)

    # Create a blank region of the original image
    all_cells = np.zeros(cell_droplet.shape)

    # First with no cells
    for index, prop in enumerate(cell_droplet_props):
        # If the region properties are within the threshold
        if 1500 <= prop.area:
            if prop.area <= 10000 and prop.extent > .2:
                # Select the region
                temp_seg = cell_droplet_labels == index + 1
                filled_seg = temp_seg
                # Add to the blank image
                all_cells = all_cells + filled_seg

    return all_cells


def cell_bright_gfp_thresh(droplet_label, droplet_props, bright_file, gfp_file):
    """
    Use the threshold droplet labels and properties in combination with
    the brightfield and GFP image to return output of segmented cell and
    segmented dead cells.

    Parameters
    ----------
    droplet_label:
        numpy array where each droplet's region is a number (i.g. 1, 2, 3...)
    droplet_props:
        list of skimage.measure.properties values about droplets
    bright_file:
        filename of the brightfield microscopy image
        (same dimensions of droplet_label and gfp_file)
    gfp_file:
        filename of the gfp microscopy image
        (same dimensions of droplet_label and bright_file)
    """

    # Test images (brightfield and GFP)
    test_raw_image = skimage.io.imread(bright_file)[:, :, 0]
    test_gfp_image = skimage.io.imread(gfp_file)[:, :, 1]
    # Actual droplet segmented GFP image properties
    gfp_droplet_props = skimage.measure.regionprops(droplet_label,
                                                    test_gfp_image)

    # Create list of Segmented GFP droplet images
    gfp_droplets = []
    for index, prop in enumerate(gfp_droplet_props):
        gfp_droplets.append(prop.intensity_image)

    # Create list of brightfield cells
    droplet_cells_list = []
    for i in range(len(droplet_props)):
        droplet_cells_list.append(micro.cells_from_droplet(droplet_label,
                                                           test_raw_image, i))

    return droplet_cells_list, gfp_droplets
