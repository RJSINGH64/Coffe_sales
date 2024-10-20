import pandas as pd
import numpy as np
import pickle
import streamlit as st
from logger import logging 
from exception import ProjectException 
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px
import warnings
import os , sys
warnings.filterwarnings("ignore")

logging.info(f"importing dataset as dataframe")

file_path =os.path.join(os.getcwd() , 'dataset/Cofee Sales dataset.csv')
data=pd.read_csv(file_path)
logging.info(f"Rows and Columns avialable :{data.shape}")

# Handle duplicates and null values
data.drop_duplicates(inplace=True)
logging.info(f"Duplicated Values are {data.duplicated().sum().sum()}")
data.dropna(inplace=True)
logging.info(f"droping null values  {data.isnull().sum().sum()}")

# Convert date columns to datetime and extract year, month, day, and weekday
data['date'] = pd.to_datetime(data['date'])
data['year'] = data['date'].dt.year
data['month'] = data['date'].dt.month
data['day'] = data['date'].dt.day
data['hour'] = data['date'].dt.hour
data['weekday'] = data['date'].dt.weekday
logging.info(f"Creating new columns day , month , week_day and hour using datetime")

# Prepare the training data
train_df = data.drop(["date", "datetime", "cash_type"], axis=1)
logging.info(f"train df column : {list(train_df.columns)}")

# One-hot encode 'coffee_name' and 'card'
coffee_encoder = OneHotEncoder(sparse_output=False, drop="first")
encoded_coffee = coffee_encoder.fit_transform(train_df[['coffee_name']])
coffee_encoded_df = pd.DataFrame(encoded_coffee, columns=[name.replace('coffee_name_', '') for name in coffee_encoder.get_feature_names_out()])
logging.info(f"Applying one hot encoder in this column coffee_name ")
card_encoder = OneHotEncoder(sparse_output=False, drop="first")
encoded_card = card_encoder.fit_transform(train_df[['card']])
card_encoded_df = pd.DataFrame(encoded_card, columns=[name.replace('card_', '') for name in card_encoder.get_feature_names_out()])
logging.info(f"Applying one hot encoder in this column card")
train_df = pd.concat([train_df.reset_index(drop=True), coffee_encoded_df.reset_index(drop=True), card_encoded_df.reset_index(drop=True)], axis=1)
train_df = train_df.drop(['coffee_name', 'card'], axis=1)

# Define features and target
X = train_df.drop("money", axis=1)
y = train_df["money"]
logging.info(f"creating independent and dependent feature as x and y")

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
logging.info(f"Spliting x train and y train using train test split")

# Train the RandomForestRegressor and save it
model = RandomForestRegressor()
model.fit(X_train, y_train)


# Save the model and encoders
with open('artifacts/model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('artifacts/coffee_encoder.pkl', 'wb') as coffee_file:
    pickle.dump(coffee_encoder, coffee_file)

with open('artifacts/card_encoder.pkl', 'wb') as card_file:
    pickle.dump(card_encoder, card_file)

logging.info(f"All pickle files saved inside artifact folder")
logging.info(f"creating dashboard app using streamlit")
# Streamlit app configuration
st.set_page_config(page_title='Coffee Sales Prediction', layout='wide')

# Setting Streamlit theme (add this section in your Streamlit configuration)
st.markdown(
    """
    <style>
    .reportview-container {
        background: #FFEB3B; /* Yellowish background */
    }
    h1, h2, h3 {
        color: Green; /* Red headers */
        font-weight: bold; /* Bold text */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load the model and encoders
with open('artifacts/model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('artifacts/coffee_encoder.pkl', 'rb') as coffee_file:
    coffee_encoder = pickle.load(coffee_file)

with open('artifacts/card_encoder.pkl', 'rb') as card_file:
    card_encoder = pickle.load(card_file)
logging.info(f"loading pickle files........")

# Sidebar for user input
st.sidebar.title('Coffee Sales App')
st.sidebar.markdown("""
<style>
.sidebar .sidebar-content {
    background-color: #4CAF50; /* Green sidebar */
    color: #FF0000; /* Red text */
}
</style>
""", unsafe_allow_html=True)

options = st.sidebar.radio('Select Option:', ['Prediction', 'Analysis'])

logging.info(f"prediction started ")
# Prediction Section
if options == 'Prediction':
    st.title('**Coffee Sales Prediction**')
    st.write("**Fill in the details below to predict coffee sales.**")

    # User input for prediction
    selected_coffee = st.selectbox('**Select Coffee**', options=coffee_encoded_df.columns.tolist())
    selected_card = st.selectbox('**Select Card**', options=card_encoded_df.columns.tolist())
    selected_month = st.selectbox('**Select Month**', options=data['month'].unique())
    selected_hour = st.slider('**Select Hour**', min_value=0, max_value=23)
    selected_weekday = st.selectbox('**Select Weekday**', options=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

    # Map weekday to numerical value
    weekday_mapping = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
    weekday_num = weekday_mapping[selected_weekday]

    # Prepare the input data for prediction
    input_data = np.zeros(X.shape[1])
    input_data[np.where(X.columns.str.contains(selected_coffee))] = 1
    input_data[np.where(X.columns.str.contains(selected_card))] = 1
    input_data[X.columns.get_loc('month')] = selected_month
    input_data[X.columns.get_loc('hour')] = selected_hour
    input_data[X.columns.get_loc('weekday')] = weekday_num
    
    if st.button('**Predict Sales Price**'):
        prediction = model.predict([input_data])
        st.success(f'The predicted sales price is: {prediction[0]:.2f}')  # No dollar sign
        logging.info(f"Predicted Sale Price for {selected_coffee} is  {prediction[0]:.2f}")
# Analysis Section
elif options == 'Analysis':
    logging.info(f"data visualization started")
    st.title('**Coffee Sales Analysis Dashboard**')
    st.write("**Explore the coffee sales data through visualizations.**")

    # Example analysis: Distribution of sales
    st.subheader('Sales Distribution')
    fig = px.histogram(data, x='money', nbins=30, title='Distribution of Coffee Sales Prices')
    st.plotly_chart(fig)

    # Example analysis: Sales by Coffee Type
    st.subheader('Sales by Coffee Type')
    sales_by_coffee = data.groupby('coffee_name')['money'].sum().reset_index()
    fig2 = px.bar(sales_by_coffee, x='coffee_name', y='money', title='Total Sales by Coffee Type', color='money')
    st.plotly_chart(fig2)

    # Add more analysis options here as needed
    logging.info(f" data plotted sucessfully")