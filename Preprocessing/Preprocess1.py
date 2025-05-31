import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.compose import ColumnTransformer


df = pd.read_csv("../datasets/creditcard.csv")

# Handling Missing Values (mean imputation for all numeric columns)
num_cols = df.select_dtypes(include=["float64", "int64"]).columns
num_imputer = SimpleImputer(strategy="mean")
df[num_cols] = num_imputer.fit_transform(df[num_cols])

# Normalization
scaler = MinMaxScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# Encoding Categorical Variables
# Identify categorical columns
cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

threshold = 10
high_card = [c for c in cat_cols if df[c].nunique() > threshold]
low_card  = [c for c in cat_cols if df[c].nunique() <= threshold]

# Label encode high cardinality columns
for col in high_card:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

# One-hot encode low cardinality columns
df = pd.get_dummies(df, columns=low_card, drop_first=True)
