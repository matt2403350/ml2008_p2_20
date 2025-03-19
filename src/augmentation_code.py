import os
import random
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

folder_path = 'src/SEA_IMG'
augmented_folder_path = 'src/SEA_IMG_augmented'

#Created augmented folder
if not os.path.exists(augmented_folder_path):
    os.makedirs(augmented_folder_path)

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

# Clone folder structure and copy images for data preprocessing
def clone_folder_structure(src_folder, dest_folder):
    for subfolder in os.listdir(src_folder):
        src_subfolder_path = os.path.join(src_folder, subfolder)
        dest_subfolder_path = os.path.join(dest_folder, subfolder)

        if os.path.isdir(src_subfolder_path):
            #Create coressponding subfolders into augmented folder
            if not os.path.exists(dest_subfolder_path):
                os.makedirs(dest_subfolder_path)

            #Copy images over
            for file in os.listdir(src_subfolder_path):
                if file.endswith('.jpg'):
                    src_file_path = os.path.join(src_subfolder_path, file)
                    dest_file_path = os.path.join(dest_subfolder_path, file)
                    if not os.path.exists(dest_file_path):
                        shutil.copy(src_file_path, dest_file_path)
                    

# Augmentation / Addition of images
def augment_images(subfolder_path, target_count=449):
    files = os.listdir(subfolder_path)
    image_files = [file for file in files if file.endswith('.jpg')]
    total_images = len(image_files)

    # Check if the number of images in the subfolder is less than 449
    if total_images < target_count:
        print(f"Subfolder '{os.path.basename(subfolder_path)}' contains less than 449 images. Augmenting images...")

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
                if i >= 10 or total_images >= target_count:  # Generate up to 10 augmented images per original image
                    break

        print(f"Subfolder '{os.path.basename(subfolder_path)}' now contains {total_images} images.")

    else:
        print(f"Subfolder '{os.path.basename(subfolder_path)}' already contains 449 images. No augmentation needed.")


# Deletion of images
def delete_images(subfolder_path, target_count=449):
        files = os.listdir(subfolder_path)
        image_files = [file for file in files if file.endswith('.jpg')]
        total_images = len(image_files)

        # Check if the number of images in the subfolder is more than 449
        if total_images > target_count:
            print(f"Subfolder '{os.path.basename(subfolder_path)}' contains {total_images} images. Performing undersampling...")

            # Calculate the number of images to delete
            num_to_delete = total_images - target_count

            # Randomly select images to delete
            images_to_delete = random.sample(image_files, num_to_delete)

            # Delete the selected images
            for image_file in images_to_delete:
                image_path = os.path.join(subfolder_path, image_file)
                try:
                    os.remove(image_path)
                except Exception as e:
                    print(f"Failed to delete {image_path}: {e}")

            print(f"Subfolder '{os.path.basename(subfolder_path)}' now contains 449 images.")
        elif total_images == 449:
            print(f"Subfolder '{os.path.basename(subfolder_path)}' already contains 449 images. No deletion needed.")
        else:
            print(f"Subfolder '{os.path.basename(subfolder_path)}' contains less than 449 images. No deletion needed.")

#RUN CODE HERE
# Clone folder structure and copy images for data preprocessing
clone_folder_structure(folder_path, augmented_folder_path)

#Process augmentation and deletion
for subfolder in os.listdir(augmented_folder_path):
    subfolder_path = os.path.join(augmented_folder_path, subfolder)
    if os.path.isdir(subfolder_path):
        augment_images(subfolder_path)
        delete_images(subfolder_path)
