import matplotlib.pyplot as plt
from matplotlib import offsetbox
import hashlib
from PIL import Image
import numpy as np

def plot_components(data, model, images = None, ax = None, thumb_frac = 0.05,
                    cmap = 'gray_r', prefit = False, zoom = 0.1):
    
    ax = ax or plt.gca()

    """
    simple scatterplot of the data
    """
    ax.scatter(data[:, 0], data[:, 1], c = cmap)
    """
    if images are included with this function, they are included here and are
    resized for better visualization of the whole sapce.
    """
    if images is not None:
        """
        creates a maximum amount of permitted overlap of the images, controlled
        by the thumb_frac variable, which influences how far apart images should
        be to avoid overlap.
        """
        min_dist_2 = (thumb_frac * max(data.max(0) - data.min(0))) ** 2
        """
        creates a dummy image placement position outside the boundaries of plot
        to ensure that an image will always end up being placed.
        """
        shown_images = np.array([2 * data.max(0)]) 
        
        for i in range(data.shape[0]):
            """
            this calculates the next possible position for an image and if it is
            further away than the min_dist, then it is placed.
            """
            dist = np.sum((data[i] - shown_images) ** 2, 1)
            if np.min(dist) < min_dist_2:
                continue

            """
            adds the current image (if eligable) to the list of images to plot.
            """
            shown_images = np.vstack([shown_images, data[i]])
            x, y = data[i][:2]
            """
            this uses the offsetbox function to place specific images for each
            item on specific scatterplot points that match the same class.
            """
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(images[i], zoom = zoom),
                (x, y) 
            )
            ax.add_artist(imagebox)

def compute_image_hash(img):
    """Compute SHA-256 hash of an image"""
    hasher = hashlib.sha256()
    # convert image to a consistent format
    img = img.convert("RGB")
    """
    resize to a consistent size (optional for exact matching, but
    can improve robustness)
    """
    img = img.resize((256, 256))
    # update hash with image data
    hasher.update(img.tobytes())
    return hasher.hexdigest()

def image_processing(df):
    """
    this takes the image id and product id, and the directory and desired save directory and then opens each image, one by one
    (if used with .apply(axis = 1)) and converts it into an np.array.
    """
    image_name = df['image']
    image_directory = 'data/images/image_train/'
    image_path = image_directory + image_name

    with Image.open(image_path) as img:
        img_array = np.array(img)
    """
    this is what filters out the color specifications. I lowered it as much as a could for good results while still
    being sensitive to some light yellows and blues.
    """
    locations = np.where(np.all(img_array <= [248, 235, 235], axis = -1))
    """
    sometimes the image locations returns as a 1D array when the full image is an image, in that case
    resize to desired shape and save it.
    """
    if len(locations[0]) == 0 or len(locations[1]) == 0:
        pil_image = Image.fromarray(img_array)
        new_pil_image = pil_image.resize((300, 300), Image.LANCZOS)
        img = np.array(new_pil_image)
        ratio = 1
        ratios = 1
        coords = [0, 500, 0, 500]
        xycoords = [[0, 0, 500, 500, 0], [0, 500, 500, 0, 0]]
    
        return image_name, ratio, coords, xycoords, img

    else:    
        coords = [np.min(locations[1]), np.max(locations[1]), np.min(locations[0]), np.max(locations[0])]
        #area = (coords[1] - coords[0]) * (coords[3] - coords[2])        
        xycoords = [[coords[0], coords[0], coords[1], coords[1], coords[0]], [coords[2], coords[3], coords[3], coords[2], coords[2]]]
    
        image2 = img_array[coords[2]:coords[3], coords[0]:coords[1]]
        pil_image = Image.fromarray(image2)
        ratio = (pil_image.width * pil_image.height) / (500 * 500)
        new_width, new_height = 300, 300
        aspect_ratio = pil_image.width / pil_image.height
        """
        selects operations based on the largest size and then calculates the relative increase needed for the
        smaller size to maintain ratios while having the largest size at 300 pixels. White padding is then
        added on the top/bottom or left/right in equal preportions. Finally, the image is saved.
        """
        if pil_image.width > pil_image.height:
            new_height = int(new_width / aspect_ratio)
            resized_image_pil = pil_image.resize((new_width, new_height), Image.LANCZOS)
            
            if resized_image_pil.height < 300:
                value = 300 - resized_image_pil.height
                half2 = value // 2
                half1 = value - half2
                top = np.full((half1, 300, 3), fill_value = [255, 255, 255], dtype = np.uint8)
                bottom = np.full((half2, 300, 3), fill_value = [255, 255, 255], dtype = np.uint8)
                resized_image_np = np.array(resized_image_pil)
                new_image = np.concatenate((top, resized_image_np, bottom), axis = 0)
            else:
                new_image = np.array(resized_image_pil)
        
        else:
            new_width = int(new_height * aspect_ratio)
            resized_image_pil = pil_image.resize((new_width, new_height), Image.LANCZOS)
            
            if resized_image_pil.width < 300:
                value = 300 - resized_image_pil.width
                half2 = value // 2
                half1 = value - half2
                left = np.full((300, half1, 3), fill_value = [255, 255, 255], dtype = np.uint8)
                right = np.full((300, half2, 3), fill_value = [255, 255, 255], dtype = np.uint8)
                resized_image_np = np.array(resized_image_pil)
                new_image = np.concatenate((left, resized_image_np, right), axis = 1)
            else:
                new_image = np.array(resized_image_pil)

    """
    returns the image_name, ratio, aspect_ratio, coords, and xycoords for each plotting or analysis of the shapes of the images without the white borders.
    """
    return image_name, ratio, aspect_ratio, coords, xycoords, new_image
