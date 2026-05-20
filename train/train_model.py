import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import SMOTE  # 🔥 NEW

from preprocess import preprocess_data

# Load
df = pd.read_csv("data/loan_data.csv")

# Preprocess
df = preprocess_data(df)

# One-hot encoding
cat_cols = [
    "person_gender",
    "person_education",
    "person_home_ownership",
    "loan_intent",
    "previous_loan_defaults_on_file"
]

ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
encoded = ohe.fit_transform(df[cat_cols])
encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(cat_cols))

df = pd.concat([df.drop(columns=cat_cols).reset_index(drop=True), encoded_df], axis=1)

# Split
X = df.drop("loan_status", axis=1)
y = df["loan_status"]

# 🔥 HANDLE IMBALANCE
smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 🔥 BETTER MODEL
model = RandomForestClassifier(n_estimators=150, max_depth=5,random_state=42)
model.fit(X_train_scaled, y_train)

# Accuracy
y_pred = model.predict(X_test_scaled)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))

# Save
pickle.dump(model, open("model/model.pkl", "wb"))
pickle.dump(scaler, open("model/scaler.pkl", "wb"))
pickle.dump(ohe, open("model/ohe.pkl", "wb"))
pickle.dump(X.columns, open("model/columns.pkl", "wb"))

print("🚀 Improved model trained!")