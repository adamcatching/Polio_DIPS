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

        # If the image has multiple channels, choose the channel to determine droplets from
        if self.multi_channel:
            image_bright = (self.image)[:, :, bright_channel]
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
                num_x_ticks = (image_bright.shape)[1] // mm + 1
                num_y_ticks = (image_bright.shape)[0] // mm + 1
                plt.imshow(image_droplets)
                plt.xticks(np.arange(0, (image_bright.shape)[1], mm), np.arange(0, num_x_ticks))
                plt.yticks(np.arange(0, (image_bright.shape)[0], mm), np.arange(0, num_y_ticks))
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

    def find_brightfield_cells(self):
        """Return the brightfield cells"""

        # Call the labeled black-white droplet image and the associated properties
        image_labeled, image_props = self.droplet_segment()

        # Define the droplet images
        labeled_droplets, number_droplets = skimage.measure.label(image_labeled, background=0, return_num=True)
        bright_droplet_props = skimage.measure.regionprops(labeled_droplets, image_bright)

        return labeled_droplets, bright_droplet_props