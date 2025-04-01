import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Sample DataFrame
data = { 'EmployeeID': [1, 2, 3, 4],
         'Department': ['HR', 'Finance', 'IT', 'Marketing'],
         'Salary': [60000, 70000, 80000, 90000] }

df = pd.DataFrame(data)
print(f"Original DataFrame:\n{df}\n")

# Extract the categorical column
categorical_column = df.select_dtypes(include=['object']).columns.tolist()

# Initialize OneHotEncoder
encoder = OneHotEncoder(sparse_output=False)

# Apply OneHotEncoder to the categorical column
encoded_data = encoder.fit_transform(df[categorical_column])

encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_column))

# Concatenate the original DataFrame with the encoded DataFrame
df_encoded = pd.concat([df.drop(columns=categorical_column), encoded_df], axis=1)
print(f"DataFrame after One-Hot Encoding:\n{df_encoded}\n")
# Display the columns of the new DataFrame
print(f"Columns of the new DataFrame:\n{df_encoded.columns.tolist()}\n")
# Display the first few rows of the new DataFrame
print(f"First few rows of the new DataFrame:\n{df_encoded.head()}\n")
# Display the shape of the new DataFrame
print(f"Shape of the new DataFrame: {df_encoded.shape}\n")

