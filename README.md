# DOTNET

## Overview
This project utilizes GMNS (General Modeling Network Specification) datasets for transportation network analysis. Please follow the instructions below carefully to set up the data and environment before running the analysis.

## 🛠️ Installation & Environment

To set up the project environment and dependencies, please refer to the **[INSTALLATION.md](INSTALLATION.md)** file included in this repository.

## 📂 Data Setup (Required)

Before running any code, you must manually download the dataset and place it in the correct directory.

1.  Create a folder named `data` at the root of the project.
2.  Download the **Milwaukee** dataset files from the following link:
    [GMNS_Plus_Dataset - Milwaukee (GitHub)](https://github.com/HanZhengIntelliTransport/GMNS_Plus_Dataset/tree/main/28_Milwaukee)
3.  Place the downloaded files inside the `/data/` folder.

Your file structure should look like this:

DOTNET/

├── data/               <-- Create this folder and put downloaded files here

├── data_preprocessed/  <-- Preprocessed datas will appear here after running the preprocessing notebook 

├── GMNS_Tools/         <-- Do not modify

├── notebook/           <-- Jupyter Notebooks

├── src/                <-- Do not modify

├── results/            <-- Generated results will appear here

├── .gitignore          <-- Do not modify

├── INSTALLATION.md

└── README.md

## 🚀 Usage

**Important:** There is a strict order of operations for running the notebooks.

### Step 1: Preprocessing
Navigate to the `notebook` directory and open **`preprocessing.ipynb`**.
You **must** execute **"Run All"** in this notebook to prepare the data for the rest of the project.

> ⚠️ **Warning:** Do not skip this step. The other notebooks rely on the outputs generated here.

### Step 2: Analysis
Once the preprocessing is complete, you may proceed to open and run the other notebooks located in the `notebook` directory.

