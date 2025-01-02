import random
import matplotlib.pyplot as plt
from PIL import Image

def plot_transformed_images(image_paths, labels, transform, n=3, seed=None):
    # Use the seed if mentioned
    if seed:
        random.seed(seed)

    # Create a tuple of image paths and the respective labels of the images
    data = list(zip(image_paths, labels))

    # Select random images
    random_image_paths = random.sample(data, k=n)
    for image in random_image_paths: # iterate through each image 
        img, y = image               # unpack the image and label
        with Image.open(img) as f:
            # Create a subplot of 2 images one for original and other for transformed image 
            fig, ax = plt.subplots(1, 2)

            # Visualize the original Image
            ax[0].imshow(f) 
            ax[0].set_title(f"Original \nSize: {f.size}")
            ax[0].axis("off")

            # Visualize the transformed image
            transformed_image = transform(f).permute(1, 2, 0) 
            ax[1].imshow(transformed_image) 
            ax[1].set_title(f"Transformed \nSize: {transformed_image.shape}")
            ax[1].axis("off")

            # Mention the respective class of the image
            fig.suptitle(f"Class: {y}", fontsize=16)