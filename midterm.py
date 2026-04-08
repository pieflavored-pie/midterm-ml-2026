# dataset from https://www.kaggle.com/datasets/nitikachandel95/online-learning-engagement-and-performance-oulad

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("online_education_dataset.csv")
df.drop(["id_student", "gender", "region", "engagement_level", "performance_level", "risk_level", "dropout_flag", "final_result"], axis=1)

"""
processing numeric features that have nulls: total_clicks, avg_score:
fill with 0 since those students did not participate in the courses
"""
fillna_val = {"total_clicks": 0, "avg_score": 0}
df.fillna(value=fillna_val, inplace=True)

"""
processing categorical features: highest_education, imd_band
1. highest_education: ordinal encoding (education level from low to high)
2. imd_band: students from North Region and Ireland don't have IMD (probably the gov doesn't include those regions)
- create a column acting as an indicator: IMD missing or not
- then apply ordinal encoding, filling nulls with median, assuming average
- why median imputation?
  + https://vtiya.medium.com/when-to-use-mean-median-mode-imputation-b0fd6be247db
"""
# 1. highest_education
# uniq_edu_lvl = np.unique(df["highest_education"])
edu_lvl_map = {
    "No Formal quals": 1, "Lower Than A Level": 2, "A Level or Equivalent": 3,
    "HE Qualification": 4, "Post Graduate Qualification": 5
}
df["highest_education_numeric"] = df["highest_education"].map(edu_lvl_map)

# 2. imd_band
# null value gets 1, everything else gets 0
df["imd_missing"] = df["imd_band"].isnull().astype(int)

# ordinal encoding + filling nulls
imd_map = {
    '0-10%': 1, '10-20': 2, '20-30%': 3, '30-40%': 4, '40-50%': 5,
    '50-60%': 6, '60-70%': 7, '70-80%': 8, '80-90%': 9, '90-100%': 10
}
df["imd_band_numeric"] = df["imd_band"].map(imd_map)
df["imd_band_numeric"] = df["imd_band_numeric"].fillna(df["imd_band_numeric"].median())

"""
splitting dataset:
1. convert to numpy array
2. split into training, validation, and test (60/20/20)
"""
# X: matrix containing all features
X = df[["highest_education_numeric", "studied_credits", "imd_band_numeric", "imd_missing", "total_clicks", "avg_score"]].to_numpy()

# Y: column vector containing targets
Y = df[["pass_flag"]].to_numpy()

# 1. split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=43)
# 2. split X_train into X_train and X_val
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=43)
# use X_train to train, X_val to validate, X_test to test
