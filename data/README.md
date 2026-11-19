# Dataset — FedGraph-Diabetes

## Required Dataset

Place the CDC BRFSS 2015-derived Diabetes Health Indicators dataset here as:

```
data/diabetes.csv
```

## Dataset Details

- **Name:** Diabetes Health Indicators Dataset (CDC BRFSS 2015)
- **Records:** 70,692
- **Features:** 21 health indicator features + 1 target column
- **Target:** `Diabetes_binary` (0 = No diabetes, 1 = Diabetes/Prediabetes)
- **Balance:** 50/50 split between diabetic and non-diabetic records

## Download

Download from Kaggle:
https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset

File to download: `diabetes_binary_5050split_health_indicators_BRFSS2015.csv`

Rename it to `diabetes.csv` and place it here.

## Required Columns

The CSV must contain these exact columns:

```
Diabetes_binary, HighBP, HighChol, CholCheck, BMI, Smoker, Stroke,
HeartDiseaseorAttack, PhysActivity, Fruits, Veggies, HvyAlcoholConsump,
AnyHealthcare, NoDocbcCost, GenHlth, MentHlth, PhysHlth, DiffWalk,
Sex, Age, Education, Income
```

## Demo Mode

If no dataset is found, the system automatically enters **DEMO MODE** using
synthetic data for UI testing. Demo mode is clearly labelled throughout the UI.

⚠️ Demo results are NOT real experimental results and should not be cited.
