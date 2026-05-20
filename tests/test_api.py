import requests

url = "http://127.0.0.1:8000/predict"

data = {
    "Applicant_Income": 5000,
    "Credit_Score": 700,
    "DTI_Ratio": 20,
    "Savings": 20000,
    "Education_Level": 1,
    "Employment_Status": "Employed",
    "Marital_Status": "Single",
    "Loan_Purpose": "Home",
    "Property_Area": "Urban",
    "Gender": "Male",
    "Employer_Category": "Private"
}

response = requests.post(url, json=data)
print(response.json())