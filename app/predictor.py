import pandas as pd
import numpy as np

def predict_loan(data, model, scaler, ohe, columns):

    # =========================
    # 🔥 BASIC INPUTS
    # =========================
    previous_emi = data.get("previous_emi", 0)
    income = data["person_income"]
    loan_ratio = data["loan_amnt"] / (income + 1)

    # =========================
    # 🔴 HARD RULE (ONLY EXTREME)
    # =========================
    if loan_ratio > 1.2:
        return {
            "prediction": "Rejected",
            "confidence": 96,
            "reason": "Loan amount is unrealistically higher than income"
        }

    if previous_emi > 0.7 * income:
        return {
            "prediction": "Rejected",
            "confidence": 97.0,
            "reason": "Extremely high EMI compared to income"
        }

    # =========================
    # 📊 CREATE DATAFRAME
    # =========================
    df = pd.DataFrame([data])

    # =========================
    # 🔥 TRANSFORM (same as training)
    # =========================
    df["person_income"] = np.log1p(df["person_income"])
    df["loan_amnt"] = np.log1p(df["loan_amnt"])

    # =========================
    # 🔥 FEATURE ENGINEERING
    # =========================
    df["loan_income_sq"] = df["loan_percent_income"] ** 2
    df["affordability"] = df["person_income"] - df["loan_amnt"]
    df["total_obligation"] = df["loan_percent_income"] * df["person_income"]
    df["income_loan_ratio"] = df["person_income"] / (df["loan_amnt"] + 1)

    # =========================
    # 🔤 ENCODING
    # =========================
    cat_cols = [
        "person_gender",
        "person_education",
        "person_home_ownership",
        "loan_intent",
        "previous_loan_defaults_on_file"
    ]

    encoded = ohe.transform(df[cat_cols])
    encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(cat_cols))

    df = pd.concat(
        [df.drop(columns=cat_cols).reset_index(drop=True), encoded_df],
        axis=1
    )

    # =========================
    # 📏 MATCH TRAINING COLUMNS
    # =========================
    df = df.reindex(columns=columns, fill_value=0)

    # =========================
    # ⚖️ SCALE
    # =========================
    df_scaled = scaler.transform(df)

    # =========================
    # 🤖 MODEL PREDICTION
    # =========================
    pred = model.predict(df_scaled)[0]
    prob = model.predict_proba(df_scaled)[0][1]

    # =========================
    # 🧠 RISK SCORING (SMART LOGIC)
    # =========================
    risk = 0
    reasons = []

    credit = data["credit_score"]
    interest = data["loan_int_rate"]
    exp = data["person_emp_exp"]
    history = data["cb_person_cred_hist_length"]
    default = data["previous_loan_defaults_on_file"]

    # 🔴 NEGATIVE FACTORS
    if credit < 600:
        risk += 2
        reasons.append("Low credit score")

    if loan_ratio > 1:
        risk += 4
        reasons.append("Loan exceeds income (very high risk)")

    elif loan_ratio > 0.7:
        risk += 3
        reasons.append("Loan too large compared to income")

    elif loan_ratio > 0.5:
        risk += 2
        reasons.append("High loan burden")

    elif loan_ratio > 0.35:
        risk += 1
        reasons.append("Moderate loan burden")

    if previous_emi > 0.5 * income:
        risk += 3
        reasons.append("High EMI burden")
    elif previous_emi > 0.3 * income:
        risk += 1
        reasons.append("Moderate EMI burden")

    if interest > 15:
        risk += 1
        reasons.append("High interest rate")

    if exp < 1:
        risk += 2
        reasons.append("Very low work experience")
    elif exp < 3:
        risk += 1
        reasons.append("Limited work experience")

    if history < 2:
        risk += 2
        reasons.append("Very short credit history")
    elif history < 5:
        risk += 1
        reasons.append("Limited credit history")

    if default == "Yes":
        risk += 3
        reasons.append("Past loan default")

    if income < 20000:
        risk += 2
        reasons.append("Low income")

    # 🟢 POSITIVE FACTORS
    positives = []


    # 🔹 Credit score
    if default == "Yes":
        if credit > 750:
            risk -= 1.5
            positives.append("Excellent credit recovery after default")
    else:
        if credit > 600:
            risk -= 1
            positives.append("Good credit score")

    # 🔹 Loan burden
    if default == "Yes":
        if loan_ratio < 0.2:
            risk -= 1.5
            positives.append("Very low loan burden despite past default")
    else:
        if loan_ratio < 0.3:
            risk -= 1
            positives.append("Low loan burden")

    # 🔹 EMI
    if previous_emi == 0:
        risk -= 1.5
        positives.append("No existing loans")

    # 🔹 Income
    if default == "Yes":
        if income > 100000:
            risk -= 1.5
            positives.append("Strong income recovery after default")
    else:
        if income > 80000:
            risk -= 1
            positives.append("Strong income")

    # 🔹 Credit history
    if default == "Yes":
        if history > 7:
            risk -= 1.5
            positives.append("Long stable credit history after default")
    else:
        if history > 5:
            risk -= 1
            positives.append("Long credit history")

    # =========================
    # 🎯 FINAL DECISION (HYBRID)
    # =========================
    if risk >= 4:
        final = "Rejected"
    elif risk <= 1:
        final = "Approved"
    else:
        final = "Approved" if pred == 1 else "Rejected"

    # =========================
    # 📝 FINAL REASON
    # =========================
    if final == "Approved":
        if positives:
            reason = "Strong profile: " + ", ".join(positives)
        else:
            reason = "Approved based on overall financial profile"
    else:
        if reasons:
            reason = "Risk factors: " + ", ".join(reasons)
        else:
            reason = "Multiple risk factors detected"

    return {
        "prediction": final,
        "confidence": round(prob * 100, 2),
        "reason": reason
    }