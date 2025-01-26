# Project Overview

This project focuses on the application of Machine Learning techniques in Water Cherenkov Detectors for Gamma and Neutron Source Classification. The analysis of data preprocessing for pure water and NaCl solution is contained in the `process_pure.ipynb` and `process_salt.ipynb` files, respectively. These notebooks handle the initial data preparation steps for both cases. On the other hand, the training and evaluation of ensemble classification models are carried out in the `voting_pure.py` and `voting_salt.py` scripts, where different model architectures and evaluation metrics are applied to assess their performance.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Files and Structure](#files-and-structure)
- [Downloading Data Files](#downloading-data-files)
- [Requirements](#requirements)
- [Usage](#usage)
- [Results and Visualizations](#results-and-visualizations)
- [Contact](#contact)

---

## Files and Structure

The repository contains the following main files and directories:

```
ale/                      # Directory containing the preconfigured Python virtual environment with all required dependencies installed. To run scripts, use the `python` command.
AmBe/                     # Directory for downloaded experimental data files
filtered_AmBe_pure/       # Directory for neutron and gamma pulses in the selected energy range for pure water
filtered_AmBe_salt/       # Directory for neutron and gamma pulses in the selected energy range for NaCl solution
preprocessed_data_pure/   # Directory for preprocessed training and testing data (balanced and scaled) for pure water
preprocessed_data_salt/   # Directory for preprocessed training and testing data (balanced and scaled) for NaCl solution
results_pure/             # Directory storing results for trained models in pure water (confusion matrices, ROC curves, etc.)
results_salt/             # Directory storing results for trained models in NaCl solution (confusion matrices, ROC curves, etc.)
models/                   # Subdirectory within results_pure/ and results_salt/ for saving trained model files
process_pure.ipynb        # Notebook for preprocessing and visualization of pure water data
process_salt.ipynb        # Notebook for preprocessing and visualization of NaCl solution data
comparison_base_classifiers.ipynb # Notebook for comparing base classifiers trained on pure water and NaCl solution
comparison_ensemble_classifiers.ipynb # Notebook for comparing ensemble classifiers trained on pure water and NaCl solution
voting_pure.py            # Script for ensemble classification using pure water data
voting_salt.py            # Script for ensemble classification using NaCl solution data
download_files.py         # Script for downloading required data files
requirements.txt          # File listing required Python dependencies
README.md                 # Project documentation
```

---

## Downloading Data Files

This project includes large data files that are not stored directly in the repository due to their size. To obtain the original files, please follow these instructions:

1. **Install `gdown`:** If you don’t have `gdown` installed, you can install it using `pip`:

    ```bash
    pip install gdown
    ```

2. **Download Files:** Run the `download_files.py` script to download the required files from Google Drive:

    ```bash
    python download_files.py
    ```

   This script will retrieve all necessary files and store them in the `AmBe/` directory. Additional files can be obtained by running the provided notebooks and scripts, and they are also listed as downloadable files.

---

## Requirements

To run this project, you have two options to set up your environment:

1. **Option 1: Activate the existing virtual environment**
   - If you wish to use the preconfigured virtual environment, first, unzip the `ale.tar.gz` file to access the environment folder:
     
     ```bash
     tar -xzvf ale.tar.gz
     ```

   - Once the environment is extracted, activate it by running the following command:

    ```bash
    source ale/bin/activate
    ```

   The required dependencies will already be installed in the virtual environment, so you don't need to install them again.

2. **Option 2: Install dependencies manually**
   - If you prefer to use your own environment, you can install the required dependencies by running the following command:

     ```bash
     pip install -r requirements.txt
     ```

---

## Usage

1. **Preprocessing Data:**
   - Open and run `process_pure.ipynb` to preprocess and visualize data for pure water.
   - Similarly, open and run `process_salt.ipynb` for the NaCl solution.

2. **Training and Evaluating Models:**
   - Use `voting_pure.py` to train and evaluate models for pure water.
   - Use `voting_salt.py` for the NaCl solution.

3. **Compare Results:**
   - Open the notebook `comparison_base_classifiers.ipynb` to compare the results of base classifiers between pure water and the NaCl solution.
   - Open the notebook `comparison_ensemble_classifiers.ipynb` to compare the results of ensemble classifiers between pure water and the NaCl solution.

Each step is self-contained, but ensure the data files are downloaded first (see [Downloading Data Files](#downloading-data-files)).

---

## Results and Visualizations

The preprocessing and analysis results for pure water and NaCl solution are detailed in their respective notebooks. Key aspects include:

- **Classes Distribution:** Visualization of class distributions before and after preprocessing to ensure balanced datasets.
- **Features Analysis:** Exploration of feature importance and distribution.
- **Correlation Matrix:** Identification of relationships between features to detect multicollinearity.
- **Preprocessing Pipeline:** Overview of data cleaning, feature scaling, and transformation steps.

### Example Visualizations
- Class distributions before and after preprocessing for both media.
- Model performance metrics such as accuracy, precision, recall, f1-score and log loss, confusion matrix, roc curves.

### Script Outputs
- The results from `voting_pure.py` and `voting_salt.py` will be stored in the `results_pure/` and `results_salt/` directories, respectively. These directories include:
  - Confusion matrices
  - ROC curves
  - Decision thresholds

- Additionally, the trained models will be saved in a `models/` subdirectory within `results_pure/` and `results_salt/`.

For a deeper analysis and specific plots, refer to the `process_pure.ipynb`, `process_salt.ipynb`, or `comparison_base_classifiers.ipynb`, `comparison_ensemble_classifiers.ipynb` notebooks.

---

## Contact

For any questions or feedback, feel free to reach out:

- **Email:** said971219@gmail.com

