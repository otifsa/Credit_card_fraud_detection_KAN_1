import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, LabelEncoder


df = pd.read_paquet("../datasets/IEEE-fraud-detection.parquet")


#All object or category-typed columns
cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()


num_cols= df.select_dtypes(include=['number']).columns.tolist()




exclude = ['TransactionID', 'isFraud']  # adjust as necessary
num_cols = [
    c for c in df.columns
    if c not in cat_cols + exclude
]

#Impute missing values 
num_imputer = SimpleImputer(strategy='mean')
df[num_cols] = num_imputer.fit_transform(df[num_cols])

# Normalize numeric columns
scaler = MinMaxScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

#Split categorical columns by cardinality for encoding
threshold = 10
high_card = [c for c in cat_cols if df[c].nunique() > threshold]
low_card  = [c for c in cat_cols if df[c].nunique() <= threshold]

# Label-encode high-cardinality categories
for col in high_card:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

#One-hot encode low-cardinality categories
df = pd.get_dummies(df, columns=low_card, drop_first=True)