# Project Overview

This project focuses on the application of Machine Learning techniques in Water Cherenkov Detectors. The analysis of data preprocessing for pure water and NaCl solution is contained in the `process_pure.ipynb` and `process_salt.ipynb` files, respectively. These notebooks handle the initial data preparation steps for both cases. On the other hand, the training and evaluation of classification models are carried out in the `voting_pure.py` and `voting_salt.py` scripts, where different model architectures and evaluation metrics are applied to assess their performance.

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

## Requirements

To run this project, you need to install several dependencies. You can easily do this by using the `requirements.txt` file included in the repository. Run the following command in your terminal:

```bash
pip install -r requirements.txt
```

## Contact

**Email** said971219@gmail.com

