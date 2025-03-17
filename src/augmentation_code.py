import os
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

folder_path = 'SEA_IMG'

# Initialize the ImageDataGenerator
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Iterate through each subfolder in the main folder
for subfolder in os.listdir(folder_path):
    subfolder_path = os.path.join(folder_path, subfolder)
    if os.path.isdir(subfolder_path):
        files = os.listdir(subfolder_path)
        image_files = [file for file in files if file.endswith('.jpg')]
        total_images = len(image_files)

        # Check if the number of images in the subfolder is less than 449
        if total_images < 449:
            print(f"Subfolder '{subfolder}' contains less than 449 images. Augmenting images...")

            # Augment images until the total reaches 449
            for image_file in image_files:
                if total_images >= 449:
                    break

                image_path = os.path.join(subfolder_path, image_file)
                img = load_img(image_path)  # Load the image
                x = img_to_array(img)  # Convert the image to a numpy array
                x = x.reshape((1,) + x.shape)  # Reshape the array

                # Generate augmented images and save them
                i = 0
                for batch in datagen.flow(x, batch_size=1, save_to_dir=subfolder_path, save_prefix='aug', save_format='jpg'):
                    i += 1
                    total_images += 1  # Update the total number of images
                    if i >= 10 or total_images >= 449:  # Generate up to 10 augmented images per original image
                        break

            print(f"Subfolder '{subfolder}' now contains {total_images} images.")


# Iterate through each subfolder in the main folder
for subfolder in os.listdir(folder_path):
    subfolder_path = os.path.join(folder_path, subfolder)
    if os.path.isdir(subfolder_path):
        files = os.listdir(subfolder_path)
        image_files = [file for file in files if file.endswith('.jpg')]
        total_images = len(image_files)

        # Check if the number of images in the subfolder is more than 449
        if total_images > 449:
            print(f"Subfolder '{subfolder}' contains {total_images} images. Performing undersampling...")

            # Calculate the number of images to delete
            num_to_delete = total_images - 449

            # Randomly select images to delete
            images_to_delete = random.sample(image_files, num_to_delete)

            # Delete the selected images
            for image_file in images_to_delete:
                image_path = os.path.join(subfolder_path, image_file)
                try:
                    os.remove(image_path)
                    
                except Exception as e:
                    print(f"Failed to delete {image_path}: {e}")

            print(f"Subfolder '{subfolder}' now contains 449 images.")
        elif total_images == 449:
            print(f"Subfolder '{subfolder}' already contains 449 images. Skipping...")
        else:
            print(f"Subfolder '{subfolder}' contains less than 449 images. No action taken.")