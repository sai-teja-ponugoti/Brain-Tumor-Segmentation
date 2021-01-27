import random
from PIL import Image
import matplotlib.pyplot as plt

def displayImageMask(image_paths, masks_paths, num_image_pairs):
    print("Displaying random image and mask pairs")
    for i in random.sample(range(0, len(image_paths)), num_image_pairs):
        #         print(image_paths["images"][i])
        #         print(masks_paths["masks"][i])
        image = Image.open(image_paths["images"][i])
        mask = Image.open(masks_paths["masks"][i])
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 10))
        ax1.imshow(image)
        ax2.imshow(mask)

