import pandas as pd
import arff
import os

def save_to_arff(df, arff_file, relation_name="dataset"):
    """
    Save a pandas DataFrame to an ARFF file.

    :param df: Pandas DataFrame
    :param arff_file: Output ARFF file path
    :param relation_name: Relation name for ARFF
    """
    arff_data = {
        'description': '',
        'relation': relation_name,
        'attributes': [
            (col, 'NUMERIC') if pd.api.types.is_numeric_dtype(df[col]) else (col, list(map(str, df[col].dropna().unique())))
            for col in df.columns
        ],
        'data': df.values.tolist()
    }
    with open(arff_file, 'w') as f:
        arff.dump(arff_data, f)

# Create output directory for ARFF files
output_dir = "arff_files"
os.makedirs(output_dir, exist_ok=True)

# Dataset 1: Liver Disorders (BUPA)
liver_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/liver-disorders/bupa.data"
liver_columns = ["mcv", "alkphos", "sgpt", "sgot", "gammagt", "drinks", "selector"]
liver_data = pd.read_csv(liver_url, header=None, names=liver_columns)
# Retain categorical nature of 'selector'
save_to_arff(liver_data, os.path.join(output_dir, "liver_disorders.arff"), relation_name="liver_disorders")

# Dataset 2: Pima Indians Diabetes
pima_url = "https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database"
pima_columns = ["preg", "glucose", "bp", "skin", "insulin", "bmi", "dpf", "age", "class"]
pima_data = pd.read_csv("diabetes.csv", names=pima_columns, skiprows=1)  # Download manually if needed
# Retain 'class' as categorical
save_to_arff(pima_data, os.path.join(output_dir, "pima_diabetes.arff"), relation_name="pima_diabetes")

# Dataset 3: Wisconsin Diagnostic Breast Cancer (WDBC)
wdbc_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
wdbc_columns = [
    "ID", "Diagnosis",
    "Radius_Mean", "Texture_Mean", "Perimeter_Mean", "Area_Mean", "Smoothness_Mean",
    "Compactness_Mean", "Concavity_Mean", "Concave_Points_Mean", "Symmetry_Mean", "Fractal_Dimension_Mean",
    "Radius_SE", "Texture_SE", "Perimeter_SE", "Area_SE", "Smoothness_SE",
    "Compactness_SE", "Concavity_SE", "Concave_Points_SE", "Symmetry_SE", "Fractal_Dimension_SE",
    "Radius_Worst", "Texture_Worst", "Perimeter_Worst", "Area_Worst", "Smoothness_Worst",
    "Compactness_Worst", "Concavity_Worst", "Concave_Points_Worst", "Symmetry_Worst", "Fractal_Dimension_Worst"
]
wdbc_data = pd.read_csv(wdbc_url, header=None, names=wdbc_columns)
wdbc_data = wdbc_data.drop(columns=["ID"])  # Drop 'ID'
# Retain 'Diagnosis' as categorical
save_to_arff(wdbc_data, os.path.join(output_dir, "wdbc.arff"), relation_name="wdbc")

# Dataset 4: Heart Disease (Cleveland)
heart_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
heart_columns = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
]
heart_data = pd.read_csv(heart_url, header=None, names=heart_columns, na_values="?")
# Retain 'sex', 'cp', 'restecg', 'slope', 'ca', 'thal', and 'target' as categorical
save_to_arff(heart_data, os.path.join(output_dir, "heart_disease.arff"), relation_name="heart_disease")

# Dataset 5: Vehicle Silhouette
vehicle_dir = "statlog+vehicle+silhouettes"  # Assuming the files are downloaded manually
vehicle_dat_files = [file for file in os.listdir(vehicle_dir) if file.endswith(".dat")]
vehicle_dataframes = []
for file in sorted(vehicle_dat_files):
    file_path = os.path.join(vehicle_dir, file)
    df = pd.read_csv(file_path, header=None, delim_whitespace=True)
    vehicle_dataframes.append(df)
vehicle_data = pd.concat(vehicle_dataframes, ignore_index=True)
vehicle_columns = [
    "COMPACTNESS", "CIRCULARITY", "DISTANCE_CIRCULARITY", "RADIUS_RATIO", 
    "PR.AXIS_ASPECT_RATIO", "MAX.LENGTH_ASPECT_RATIO", "SCATTER_RATIO", 
    "ELONGATEDNESS", "PR.AXIS_RECTANGULARITY", "MAX.LENGTH_RECTANGULARITY", 
    "SCALED_VARIANCE_MAJOR", "SCALED_VARIANCE_MINOR", "SCALED_RADIUS", 
    "SKEWNESS_MAJOR", "SKEWNESS_MINOR", "KURTOSIS_MINOR", 
    "KURTOSIS_MAJOR", "HOLLOWS_RATIO", "CLASS"
]
vehicle_data.columns = vehicle_columns
# Retain 'CLASS' as categorical
save_to_arff(vehicle_data, os.path.join(output_dir, "vehicle_silhouette.arff"), relation_name="vehicle_silhouette")

print(f"ARFF files saved in '{output_dir}' directory.")