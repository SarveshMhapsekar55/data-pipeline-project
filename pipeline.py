import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# STEP 1: Load dataset
print("Loading dataset...")
data = pd.read_csv("data.csv")

print("First 5 rows of data:")
print(data.head())

# STEP 2: Separate columns
num_cols = data.select_dtypes(include=['int64', 'float64']).columns
cat_cols = data.select_dtypes(include=['object']).columns

print("\nNumerical Columns:", num_cols)
print("Categorical Columns:", cat_cols)

# STEP 3: Handle missing values

print("\nHandling missing values...")

# Numerical → mean
num_imputer = SimpleImputer(strategy='mean')
data[num_cols] = num_imputer.fit_transform(data[num_cols])

# Categorical → most frequent
cat_imputer = SimpleImputer(strategy='most_frequent')
data[cat_cols] = cat_imputer.fit_transform(data[cat_cols])

print("Missing values handled!")

# STEP 4: Encode categorical variables

print("\nEncoding categorical data...")

encoder = OneHotEncoder(sparse_output=False, drop='first')
encoded = encoder.fit_transform(data[cat_cols])

encoded_df = pd.DataFrame(
    encoded,
    columns=encoder.get_feature_names_out(cat_cols)
)

# Drop original categorical columns
data = data.drop(cat_cols, axis=1)

# Add encoded columns
data = pd.concat([data, encoded_df], axis=1)

print("Encoding complete!")

# STEP 5: Scale numerical features

print("\nScaling numerical data...")

scaler = StandardScaler()
data[num_cols] = scaler.fit_transform(data[num_cols])

print("Scaling complete!")

# STEP 6: Save processed data

print("\nSaving cleaned data...")

data.to_csv("cleaned_data.csv", index=False)

print("✅ Pipeline executed successfully!")