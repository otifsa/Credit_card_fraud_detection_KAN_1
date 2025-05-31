import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

df = pd.read_parquet("../datasets/Sparkov_data.parquet")

# Handling Missing Values 
num_cols = df.select_dtypes(include=["float64", "int64", "int32"]).columns
num_imputer = SimpleImputer(strategy="mean")
df[num_cols] = num_imputer.fit_transform(df[num_cols])

# Normalization 
scaler = MinMaxScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# Encoding Categorical Variables
# Identify categorical features 
cat_cols = [
    "merchant",
    "category",
    "gender",
    "city",
    "state",
    "job",
    "transaction_hour",
    "transaction_dayofweek",
]

# Split into high and low cardinality sets
threshold = 10
high_card   = [c for c in cat_cols if df[c].nunique() > threshold]
low_card    = [c for c in cat_cols if df[c].nunique() <= threshold]

# Label encode the high cardinality categories
for col in high_card:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

# One hot encode the low cardinality categories
df = pd.get_dummies(df, columns=low_card, drop_first=True)