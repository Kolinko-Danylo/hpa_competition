from PIL import Image # (pip install Pillow)
import numpy as np

def create_sub_masks(mask_image):
    width, height = mask_image.size

    # Initialize a dictionary of sub-masks indexed by RGB colors
    sub_masks = {}
    for x in range(width):
        for y in range(height):
            # Get the RGB values of the pixel
            pixel = mask_image.getpixel((x,y))

            # If the pixel is not black...
            if pixel != 0:
                # Check to see if we've created a sub-mask...
                pixel_str = str(pixel)
                sub_mask = sub_masks.get(pixel_str)
                if sub_mask is None:
                   # Create a sub-mask (one bit per pixel) and add to the dictionary
                    # Note: we add 1 pixel of padding in each direction
                    # because the contours module doesn't handle cases
                    # where pixels bleed to the edge of the image
                    sub_masks[pixel_str] = Image.new('1', (width+2, height+2))

                # Set the pixel value to 1 (default is 0), accounting for padding
                sub_masks[pixel_str].putpixel((x+1, y+1), 1)

    return sub_masks
import numpy as np                                 # (pip install numpy)
from skimage import measure                        # (pip install scikit-image)
from shapely.geometry import Polygon, MultiPolygon # (pip install Shapely)

def create_sub_mask_annotation(sub_mask, image_id, category_id, annotation_id, is_crowd):
    # Find contours (boundary lines) around each sub-mask
    # Note: there could be multiple contours if the object
    # is partially occluded. (E.g. an elephant behind a tree)
    sub_mask = np.array(sub_mask)
    contours = measure.find_contours(sub_mask, 0.5, positive_orientation='low')

    segmentations = []
    polygons = []
    for contour in contours:
        # Flip from (row, col) representation to (x, y)
        # and subtract the padding pixel
        for i in range(len(contour)):
            row, col = contour[i]
            contour[i] = (col - 1, row - 1)

        # Make a polygon and simplify it
        poly = Polygon(contour)
        poly = poly.simplify(1.0, preserve_topology=False)
        polygons.append(poly)
        segmentation = np.array(poly.exterior.coords).ravel().tolist()
        segmentations.append(segmentation)

    # Combine the polygons to calculate the bounding box and area
    multi_poly = MultiPolygon(polygons)
    x, y, max_x, max_y = multi_poly.bounds
    width = max_x - x
    height = max_y - y
    bbox = (x, y, width, height)
    area = multi_poly.area

    annotation = {
        'segmentation': segmentations,
        'iscrowd': is_crowd,
        'image_id': image_id,
        'category_id': category_id,
        'id': annotation_id,
        'bbox': bbox,
        'area': area
    }

    return annotation

import json
from tqdm import tqdm
import os
from concurrent.futures import ProcessPoolExecutor
# start_dir = '/common/danylokolinko/hpa_mask/hpa_cell_mask/'
# print(len(os.listdir(start_dir)))
# lst = list(map(lambda x: os.path.join(start_dir, x), lst))

if __name__ == '__main__':
    def init_process():
        global lst, is_crowd, category_id, lload

        lload = lambda filename: Image.fromarray((np.load(filename)['arr_0']))

        start_dir = '/common/danylokolinko/hpa_mask/hpa_cell_mask/'
        lst = os.listdir(start_dir)
        lst = list(map(lambda x: os.path.join(start_dir, x), lst))
        is_crowd = 0
        category_id = 1


    def process_mask(i):
        try:

            annotations = []
            image_id = i + 1
            annotation_id = 200 * image_id
            mask_path = lst[i]
            mask_image = lload(mask_path)
            sub_masks = create_sub_masks(mask_image)
            for color, sub_mask in sub_masks.items():
                annotation = create_sub_mask_annotation(sub_mask, image_id, category_id, annotation_id, is_crowd)
                annotations.append(annotation)
                annotation_id += 1
            return annotations
        except:
            return []



    ll = list(range(21806))
    # ll = list(range(21))
    with ProcessPoolExecutor(20, initializer=init_process) as executor:
        #     a = executor.map(process_mask, ll)
        results = list(tqdm(executor.map(process_mask, ll), total=len(ll)))

    with open('/common/danylokolinko/annotations.json', 'w') as f:
        json.dump(results, f)

