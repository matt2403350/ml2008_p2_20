
"""
Loads and Preprocesses the dataset/images
"""
# Import inbuilt libraries
import os, shutil, pathlib
from math import sqrt, ceil

# Import necessary libraries for data analysis
import numpy as np
import pandas as pd

# Import necessary libraries for image processing
from PIL import Image, UnidentifiedImageError

# Import necessary libraries for data visualization
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objects as go
import plotly.offline as pyo
import plotly.express as px

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Constants
img_width = 224  # Image width for preprocessing
img_height = 224  # Image height for preprocessing
random_seed = 123  # Set a fixed random seed for reproducibility
batch_size = 32  # Define the batch size for training data

working_dir = pathlib.Path().absolute()  # Path to the working directory
geo_data_dir = pathlib.Path(os.path.join(working_dir, 'src/SEA_IMG_augmented'))  # Path to the data directory

geo_train_dir = pathlib.Path(os.path.join(working_dir, 'src/train_dataset'))  # Path to the train dataset directory
geo_test_dir = pathlib.Path(os.path.join(working_dir, 'src/test_dataset'))  # Path to the test dataset directory

#print(working_dir)
#print(geo_data_dir)

# Function to extract metadata from a give file
def extract_metadata_from_folder(path):
    metadata = []
    for file_path in path.glob('*'):
        if file_path.name == '.DS_Store':
            continue  # Skip .DS_Store files
        try:
            with Image.open(file_path) as img:
                width, height = img.size
                metadata.append({
                    'country': path.name,
                    'image_name': file_path.name,
                    'width': width,
                    'height': height,
                    'size': file_path.stat().st_size,
                    'path': file_path
                })
        except UnidentifiedImageError:

            print(f"Warning: Unable to open image {file_path}")
    return metadata


# List all country folders
country_folders = [f for f in os.listdir(geo_data_dir) if os.path.isdir(os.path.join(geo_data_dir, f))]

# Initialize an empty list to store metadata
all_metadata = []

# Iterate through country folders and extract metadata
for country_name in country_folders:
    metadata = extract_metadata_from_folder(pathlib.Path(os.path.join(geo_data_dir, country_name)))
    all_metadata.extend(metadata)

# Create a Pandas DataFrame from the metadata
df_geo_data = pd.DataFrame(all_metadata)
df_data_distribution = df_geo_data.groupby('country')['image_name'].count().reset_index().rename(
    columns={'image_name': 'frequency'})
df_filtered_distribution = df_geo_data.groupby('country')['image_name'].count().reset_index().rename(
    columns={'image_name': 'frequency'})

#df_geo_data.head(10)
#df_filtered_distribution

train, test = train_test_split(df_geo_data, test_size=0.05, random_state=123)


for pathname, dataset in [
    [geo_train_dir, train],
    [geo_test_dir, test]
]:
    # make a new directory to store the filtered dataset
    pathname.mkdir(parents=True, exist_ok=True)

    # Copy the filtered dataset to the new directory
    for country in df_filtered_distribution.country.unique():
        geo_country_dir = pathname / country
        geo_country_dir.mkdir(parents=True, exist_ok=True)

        for picture_path in dataset[dataset.country == country].path.values:
            target_picture_path = geo_country_dir / picture_path.name
            shutil.copy(picture_path, target_picture_path)
