"""
This script downloads a set of files from Google Drive using their file IDs and saves them to specified directories.
Modules:
    gdown: A Python module to download files from Google Drive.
    os: Provides a way of using operating system dependent functionality.
Files:
    A list of dictionaries containing file IDs and their corresponding file paths.
Functionality:
    - Prompts the user to choose between downloading experimental data only or all files.
    - Iterates through the list of files based on the user selection.
    - For each file, checks if the directory exists; if not, creates it.
    - Constructs the download URL using the file ID.
    - Downloads the file to the specified directory.
    - Prints a message when all files have been downloaded.
Usage:
    Run this script to download the specified files from Google Drive and save them to the appropriate directories.
"""
import gdown
import os

files = [
    {"id": "1Gj9wUaT_dyvA8ECjrYRkmEZM2v3E9Ads", "name": "results_pure/models/AmBe_cdparafb.csv"},
    {"id": "1GuI8ELpt5Kz-k0pOpm3SByESFiZCeC_U", "name": "results_pure/models/AmBe_desnuda.csv"},
    {"id": "1vjSEFSmXzE2pWDjBI-gbH1XSFGAEj6sM", "name": "results_pure/models/AmBe_plomo.csv"},
    {"id": "15XY_Md9YdctsHtwUA1ctJqDaZXYIwWPU", "name": "results_pure/models/fondo.csv"},
    {"id": "1bixnzQK63uE6KWXpC2DxN7dh62rC9Aai", "name": "results_salt/models/AmBe_cdparafb.csv"},
    {"id": "1m4ZbkTGRTZTlrF0i1WVgMg5OFBxavIFQ", "name": "results_salt/models/AmBe_desnuda.csv"},
    {"id": "1c5YgDb064pQUHyNQansJs_0ZE8_71VMN", "name": "results_salt/models/AmBe_plomo.csv"},
    {"id": "1jcKmKjEsCYJEHb0P7yEV__OXq2VlSNKC", "name": "results_salt/models/fondo.csv"},
    {"id": "11dluDXbCNpsRsaDZpRfhLBk0F6u_9pqC", "name": "AmBe/AmBe_pure/AmBe_cdparafb.csv"},
    {"id": "1q1YgYnzK_kVVOTWkaypw0bNrS_kfmJmG", "name": "AmBe/AmBe_pure/AmBe_desnuda.csv"},
    {"id": "13lzL_aMNbOLArmjJPW4lysXfK1bf-WP-", "name": "AmBe/AmBe_pure/AmBe_plomo.csv"},
    {"id": "14Gtcfx9m99BmjR8hrSyvoX3EZ07wJSAQ", "name": "AmBe/AmBe_pure/fondo.csv"},
    {"id": "1AVW5vI4awA_yg-hJNs1hdzs5nPCCeBVh", "name": "AmBe/AmBe_salt/AmBe_cdparafb.csv"},
    {"id": "1hmf3aP6SMUl_C-EWdwPl_bqcbGLOr2Me", "name": "AmBe/AmBe_salt/AmBe_desnuda.csv"},
    {"id": "1EOcIvIqRkhlF_YEjbu_RczuDUacQ6On-", "name": "AmBe/AmBe_salt/AmBe_plomo.csv"},
    {"id": "1fVMo8KmiSM-MjynAYygSuYY39uT2A50t", "name": "AmBe/AmBe_salt/fondo.csv"}
]

# Filter files based on the user's selection
print("Choose an option:")
print("1. Download only experimental data (AmBe directory).")
print("2. Download all files.")
option = input("Enter 1 or 2: ").strip()

if option == "1":
    filtered_files = [file for file in files if file["name"].startswith("AmBe/")]
elif option == "2":
    filtered_files = files
else:
    print("Invalid option. Exiting.")
    exit()

# Download and create directories as needed
for file in filtered_files:
    folder_name = os.path.dirname(file["name"])  # Get the folder name
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)  # Create the folder if it doesn't exist
    
    url = f"https://drive.google.com/uc?id={file['id']}"
    output_path = os.path.join(folder_name, os.path.basename(file["name"]))  # Full path of the file
    
    # Download the file to the corresponding folder
    gdown.download(url, output_path, quiet=False)

print("All selected files have been downloaded.")
