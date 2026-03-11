import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle

# Load the trained model
model = tf.keras.models.load_model('laptop_price_prediction_model.keras')

# Load the scaler
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)
    

# Define categorical features (as used during training)
category = ['Company','TypeName','ScreenResolution','Cpu','Memory','Gpu','OpSys']

# Define numerical features (as used during training)
numerical_features = ['Inches','Ram','Weight']

# Function to get user input
def get_user_input(df_original):
    st.sidebar.header('Specify Laptop Features')

    input_data = {}
    
    # Drop 'laptop_ID' and 'Product' columns as they were dropped during training
    df_cleaned = df_original.drop(columns=['laptop_ID', 'Product'], errors='ignore')

    # Get unique values for categorical features from the original DataFrame
    for col in category:
        # Ensure we are using the original, un-encoded values for display in dropdowns
        unique_values = df_cleaned[col].unique().tolist()
        if col == 'Cpu': # For CPU, sort to improve readability
            unique_values.sort()
        elif col == 'Memory': # For Memory, sort to improve readability
            unique_values.sort()
        elif col == 'ScreenResolution': # For ScreenResolution, sort to improve readability
            unique_values.sort()
        
        input_data[col] = st.sidebar.selectbox(f'Select {col}', unique_values)

    # Get numerical input
    for col in numerical_features:
        # Use min and max from the original (unscaled) df for sliders
        min_val = float(df_original[col].min())
        max_val = float(df_original[col].max())
        default_val = float(df_original[col].mean())
        
        # Adjust step size based on range for better user experience
        if col == 'Inches':
            step = 0.1
        elif col == 'Ram':
            step = 1.0
        elif col == 'Weight':
            step = 0.01
        else:
            step = (max_val - min_val) / 100

        input_data[col] = st.sidebar.slider(f'Enter {col}', min_value=min_val, max_value=max_val, value=default_val, step=step)

    return pd.DataFrame([input_data])

# Function to preprocess user input to match training data format
def preprocess_input(user_input_df, df_original):
    # Create a dummy DataFrame with all columns present during training
    # This is to ensure that all one-hot encoded columns are present, even if not selected by user
    all_columns = df_original.drop(columns=['laptop_ID', 'Product', 'Price_euros'], errors='ignore').columns
    
    # Handle 'Ram' and 'Weight' columns before creating dummy variables
    # These were converted to int/float during training and need to be consistent
    temp_df = df_original.copy()
    if 'Ram' in temp_df.columns:
        temp_df['Ram'] = temp_df['Ram'].str.replace('GB','').astype(int)
    if 'Weight' in temp_df.columns:
        temp_df['Weight'] = temp_df['Weight'].str.replace('kg','').astype(float)

    # Re-apply one-hot encoding to original dataframe to get all possible columns
    # but drop 'Price_euros' as it's the target variable
    df_encoded_template = pd.get_dummies(temp_df.drop(columns=['Price_euros'], errors='ignore'), columns=category, drop_first=True)
    
    # Ensure boolean columns are converted to int
    for col in df_encoded_template.select_dtypes(include='bool').columns:
        df_encoded_template[col] = df_encoded_template[col].astype(int)

    # Preprocess the user input DataFrame
    # First, handle Ram and Weight if they are present as strings in user_input_df
    if 'Ram' in user_input_df.columns and isinstance(user_input_df['Ram'].iloc[0], str):
        user_input_df['Ram'] = user_input_df['Ram'].str.replace('GB','').astype(int)
    if 'Weight' in user_input_df.columns and isinstance(user_input_df['Weight'].iloc[0], str):
        user_input_df['Weight'] = user_input_df['Weight'].str.replace('kg','').astype(float)

    user_input_encoded = pd.get_dummies(user_input_df, columns=category, drop_first=True)
    
    # Align columns between user input and training data
    final_user_input = pd.DataFrame(columns=df_encoded_template.columns)
    for col in df_encoded_template.columns:
        if col in user_input_encoded.columns:
            final_user_input[col] = user_input_encoded[col]
        else:
            final_user_input[col] = 0 # Fill missing one-hot encoded columns with 0
            
    # Ensure boolean columns are converted to int
    for col in final_user_input.select_dtypes(include='bool').columns:
        final_user_input[col] = final_user_input[col].astype(int)

    # Scale numerical features
    final_user_input[numerical_features] = scaler.transform(final_user_input[numerical_features])

    return final_user_input


st.title('Laptop Price Prediction')

# Load the original dataset for reference (needed for unique values and min/max for sliders)
# Assuming 'laptop_price.csv' is in the same directory as app.py
df_original = pd.read_csv('laptop_price.csv', encoding='latin-1')

# Preprocess df_original to ensure 'Ram' and 'Weight' are numeric for get_user_input
df_original['Ram'] = df_original['Ram'].str.replace('GB','').astype(int)
df_original['Weight'] = df_original['Weight'].str.replace('kg','').astype(float)

user_input = get_user_input(df_original)

st.subheader('Your Selected Laptop Features:')
st.write(user_input)

if st.sidebar.button('Predict Price'):
    processed_input = preprocess_input(user_input, df_original)

    # Make prediction
    scaled_predicted_price = model.predict(processed_input)[0][0]
    
    # Inverse transform the scaled predicted price
    # The scaler was fit on ['Inches', 'Ram', 'Weight', 'Price_euros']
    # So we need to create a dummy array with the same shape
    dummy_array = np.zeros((1, len(numerical_features) + 1)) # +1 for Price_euros
    # Place the scaled predicted price in the position corresponding to 'Price_euros'
    price_euros_index_in_scaler = 3 # Based on numerical_features = ['Inches','Ram','Weight','Price_euros']
    dummy_array[0, price_euros_index_in_scaler] = scaled_predicted_price
    
    # Inverse transform the entire dummy array and extract the price
    original_scale_prediction = scaler.inverse_transform(dummy_array)[0, price_euros_index_in_scaler]

    st.subheader(f'Estimated Laptop Price: €{original_scale_prediction:.2f}')
