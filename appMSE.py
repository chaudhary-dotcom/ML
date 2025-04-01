import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Streamlit App title 
st.title('Linear Regression with MSE calculation')

# Generate Synthetic data or upload csv 
data_option = st.radio('Choose data source:', ('Generate Synthetic Data', 'Upload CSV'))

if data_option == 'Generate Synthetic Data':
    np.random.seed(42)
    X = 2*np.random.randn(100, 1)  
    y = 4+3*X+np.random.randn(100, 1)  # y = 4+3X + noise
    df = pd.DataFrame(np.hstack((X, y)), columns = ['X', 'y'])

else:
    uploaded_file = st.file_uploader('Upload a CSV file', type = ['csv'])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write('Preview of Uploaded Data:', df.head())
    
    else:
        st.warning('Please upload a CSV file.')
        st.stop()

# Splitting Data 
X = df.iloc[:, :-1].values   #Features (all columns eXcept last)
y = df.iloc[:, :-1].values #Target last columns

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the linear regression model 
model = LinearRegression()
model.fit(X_train, y_train)

# Predications 
y_pred = model.predict(X_test)

# Compute MSE 
mse = mean_squared_error(y_test, y_pred)
st.write(f'Mean squared error: {mse:.4f}')

# Plot actual vs predicated
fig, aX = plt.subplots()
aX.scatter(X_test, y_test, label='Actual', color ='blue')
aX.scatter(X_test, y_pred, label ='Predicated', color='red', alpha=0.7)
aX.plot(X_test, y_pred, color='green', linewidth=2)
aX.set_xlabel('X')
aX.set_ylabel('y')
aX.set_title("Actual vs predicated values")
aX.legend()
st.pyplot(fig)