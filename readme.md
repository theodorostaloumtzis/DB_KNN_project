# DB_KNN Project

This repository contains a project that integrates Python and C++ to perform dataset conversion, ARFF file generation, and K-Nearest Neighbors (KNN) classification. It demonstrates how to preprocess data, implement a KNN classifier, and evaluate it using K-Fold cross-validation.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Setup Jupyter Notebook](#setup-jupyter-notebook)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Features

- Conversion of datasets to ARFF format.
- Integration of Python and C++ using the `cppyy` library.
- K-Nearest Neighbors (KNN) implementation in C++.
- Model evaluation using K-Fold cross-validation in Python.

---

## Installation

1. **Clone the Repository:**

   Clone the GitHub repository to your local machine:

   ```bash
   git clone https://github.com/theodorostaloumtzis/DB_KNN_project.git
   cd DB_KNN_project
   ```

2. **Install System Requirements:**

   Install the necessary build tools and Python development headers:

   ```bash
   sudo apt update
   sudo apt install -y build-essential g++ python3 python3-pip python3-dev
   ```

3. **Install Python Dependencies:**

   Use the provided `requirements.txt` file to install all required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

4. **Install Jupyter Notebook:**

   Install Jupyter Notebook to run the project:

   ```bash
   pip install notebook
   ```

---

## Setup Jupyter Notebook

1. **Start Jupyter Notebook:**

   Launch the Jupyter Notebook server:

   ```bash
   jupyter notebook
   ```

   This command will open the Jupyter interface in your default web browser.

2. **Open the Notebook:**

   In the Jupyter interface, navigate to the `notebook_with_kfold.ipynb` file and open it.

3. **Verify Dependencies:**

   Before running the notebook, confirm all dependencies are installed by running:

   ```python
   !pip install -r requirements.txt
   ```

4. **Run the Notebook:**

   Execute the cells in `notebook_with_kfold.ipynb` sequentially. The notebook demonstrates dataset handling, ARFF file generation, and KNN classification.

---

## Usage

### Dataset Conversion

1. **Prepare the Datasets:**

   - Download datasets (e.g., `diabetes.csv`, `statlog+vehicle+silhouettes` folder) and place them in their respective directories.

2. **Convert Datasets to ARFF Format:**

   Run the dataset conversion script:

   ```bash
   python dataset_conversion.py
   ```

   ARFF files will be saved in the `arff_files` directory.

### Running the KNN Classifier

1. **Compile the C++ File:**

   The `functions.cpp` file is dynamically compiled using the `cppyy` library during runtime. No manual compilation is required.

2. **Run the Jupyter Notebook:**

   Follow the [Setup Jupyter Notebook](#setup-jupyter-notebook) instructions to execute the KNN classifier and evaluate the model using K-Fold cross-validation.

---

## Dependencies

### System Requirements

- **System Packages:**
  - `build-essential`
  - `g++`
  - `python3`
  - `python3-dev`

## Project Structure

```text
DB_KNN_project/
├── dataset_conversion.py          # Script for converting datasets to ARFF format
├── functions.cpp                  # C++ implementation of KNN-related functions
├── notebook_with_kfold.ipynb      # Jupyter Notebook for KNN with K-Fold cross-validation
├── requirements.txt               # Python dependencies
└── arff_files/                    # Directory for generated ARFF files
```

---

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature-name`.
3. Commit your changes: `git commit -m "Add a feature"`.
4. Push to the branch: `git push origin feature-name`.
5. Open a pull request.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Acknowledgements

- Datasets are sourced from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php) and [Kaggle](https://www.kaggle.com/).
- ARFF file handling is implemented using the `liac-arff` library.
- CPPYY is used for dynamic compilation of C++ code to improve performance of the algorithm.

---

## Contributors

- Theodoros Taloumtzis

