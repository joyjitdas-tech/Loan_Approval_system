import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

def preprocess_data(df):

    # 🔥 Remove outliers
    df = df[df["person_age"] < 80]

    # Target
    target = df["loan_status"]
    df = df.drop("loan_status", axis=1)

    # 🔥 Columns
    categorical_cols = df.select_dtypes(include=["object"]).columns
    numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns

    # 🔥 Handle missing values FIRST
    num_imp = SimpleImputer(strategy="mean")
    df[numerical_cols] = num_imp.fit_transform(df[numerical_cols])

    cat_imp = SimpleImputer(strategy="most_frequent")
    df[categorical_cols] = cat_imp.fit_transform(df[categorical_cols])

    # =========================
    # 🔥 ADD THIS (IMPORTANT)
    # =========================

    # ✅ 1. Log transformation (fix high income bug)
    df["person_income"] = np.log1p(df["person_income"])
    df["loan_amnt"] = np.log1p(df["loan_amnt"])

    # =========================
    # 🔥 FEATURE ENGINEERING
    # =========================

    df["loan_income_sq"] = df["loan_percent_income"] ** 2

    # ⚠️ Now these use transformed values
    df["affordability"] = df["person_income"] - df["loan_amnt"]

    df["total_obligation"] = df["loan_percent_income"] * df["person_income"]

    # ✅ 2. NEW normalized feature (VERY IMPORTANT)
    df["income_loan_ratio"] = df["person_income"] / (df["loan_amnt"] + 1)

    # Restore target
    df["loan_status"] = target

    return df