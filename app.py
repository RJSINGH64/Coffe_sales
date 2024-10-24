import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from logger import logging 
from exception import ProjectException 
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import plotly.express as px
import warnings
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os , sys

warnings.filterwarnings("ignore")

#Creating Directory if not avialable 
os.makedirs("artifacts" , exist_ok=True)
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

param_grid = {
    'subsample': 0.9,
    'n_estimators': 100,
    'max_depth': 5,
    'learning_rate': 0.2,
    'gamma': 0,
    'colsample_bytree': 0.8
}

# Train the XGBRegressor with the specified hyperparameters
model = XGBRegressor(**param_grid)  
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
# Set the theme
st.set_page_config(
    page_title="My Streamlit App",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Setting Streamlit theme (add this section in your Streamlit configuration)
st.markdown(
    """
    <style>
    .reportview-container {
        background: Yellow; /* Yellowish background */
    }
    h1, h2, h3 {
        color: Brown; /* Brown headers */
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

st.sidebar.image("coffee_image.png" , width=180)

# Sidebar for user input
st.sidebar.title('Coffee Sales App')

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
    #using dataset without droping duplicates
    st.title('**Coffee Sales Analysis Dashboard**')
    st.write("**Explore the coffee sales data through visualizations.**")
    
    file_path = os.path.join(os.getcwd() , "dataset/Cofee Sales dataset.csv")
    df=pd.read_csv(file_path)
    
    logging.info(f"data visualization started")
    
   # Assuming 'top_coffee_sales' is your DataFrame with the coffee sales data
    # 'top_coffee_sales' is a Series with coffee names as index and their frequencies as values
    top_coffee_sales=df["coffee_name"].value_counts()
    
    
   # Convert the 'top_coffee_sales' Series into a DataFrame for Plotly
    top_coffee_sales_df = pd.DataFrame({
           'Coffee': top_coffee_sales.index,
           'Frequency': top_coffee_sales.values
})

    # Plotting using Plotly Express
    fig = px.bar(top_coffee_sales_df, 
             x='Frequency', 
             y='Coffee', 
             orientation='h', 
             color='Coffee' ,
             title="<b>Coffee Sales by Product<b>"
             )

    # Update layout to make x and y labels bold and ensure all ticks are shown
    fig.update_layout(
        width=1200,
        height=550,
        xaxis_title="<b>Frequency</b>",
        yaxis_title="<b>Coffee</b>",
        
)
    # Display the Plotly figure in Streamlit
    st.plotly_chart(fig)

    
    
    # Example analysis: Sales by Coffee Type
    st.subheader('Sales by Coffee Type')
    cash_type = df["cash_type"].value_counts().reset_index().sort_values(by="count" , ascending=False)
    print(f"{cash_type.head()}")
    fig = px.bar(cash_type,
                  x="cash_type" , 
                  y='count' ,
                  color="cash_type",
                  title="<b>Total Payment Type  Distribution",
                  height=400)
    # Update layout to make x and y labels bold and ensure all ticks are shown
    fig.update_layout(
        width=1100,
        height=500,
        xaxis_title="<b>Cash Type</b>",
        yaxis_title="<b>Count</b>",
        
)
    
    st.plotly_chart(fig)
    
    # Assuming df is your DataFrame
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["year"] = df["date"].dt.year
    df["date_time"] = pd.to_datetime(df["datetime"])
    df["hour"] = df["date_time"].dt.hour
    df["week_day_name"] = df["date_time"].dt.day_name() 
    # Group by coffee_name and month, and sum the money
    monthly_revenue = df.groupby(['coffee_name', 'month'])['money'].sum().reset_index()

    # Pivot the data for the line plot (Seaborn)
    pivot_table = monthly_revenue.pivot(index='month', columns='coffee_name', values='money')

    # Create the line plot with Plotly
    fig1= px.line(monthly_revenue, x="month", y="money", color="coffee_name", markers=True)
    # Customize the plot
    fig1.update_layout(
    title='Monthly Coffee Sales (Plotly)',
    xaxis_title='Month',
    yaxis_title='Total Sales (Money)',
    legend_title='Coffee Name',
    width=1200, height=600
 )
    st.plotly_chart(fig1)
    
    # Sample DataFrame (replace df with your actual DataFrame)
    # Group by coffee_name to calculate total revenue for each coffee type
    coffee_revenue_df = df.groupby('coffee_name')['money'].sum().reset_index()

    # Calculate the total revenue across all coffee types
    total_revenue_all = coffee_revenue_df['money'].sum()

    # Calculate the percentage of total revenue for each coffee type
    coffee_revenue_df['percentage_of_total_revenue'] = (coffee_revenue_df['money'] / total_revenue_all) * 100

    # Display the summarized revenue data by coffee type
    st.write("### Coffee Revenue Data")
    st.dataframe(coffee_revenue_df)

    # Custom color palette
    custom_colors = ["#FF5733", "#33FF57", "#3357FF", "#FF33A1", "#FFC300", 
                 "#DAF7A6", "#900C3F", "#581845", "#FF6F61", "#C70039"]

    # Define Plotly bar chart for total revenue distribution
    fig_bar = px.bar(coffee_revenue_df, 
                 y='coffee_name', 
                 x='money', 
                 orientation='h', 
                 title='Total Revenue Distribution',
                 labels={'money': 'Total Revenue', 'coffee_name': 'Coffee Type'},
                 color_discrete_sequence=custom_colors)  # Custom colors

    # Customize the layout
    fig_bar.update_layout(
        xaxis_title="Revenue (in currency)",
        yaxis_title="Coffee Type",
        title_font=dict(size=18, family='Arial', color='black'),  # Use `font` attributes
        xaxis_title_font=dict(size=14, family='Arial', color='black'),  # Use `font` attributes
        yaxis_title_font=dict(size=14, family='Arial', color='black'),  # Use `font` attributes
)

    # Define Plotly pie chart for percentage of total revenue
    fig_pie = go.Figure(
        data=[go.Pie(
        labels=coffee_revenue_df['coffee_name'], 
        values=coffee_revenue_df['percentage_of_total_revenue'], 
        hole=.3,  # Donut chart style
        textinfo='label+percent',
        hoverinfo='label+percent+value',
        marker=dict(colors=custom_colors),  # Custom colors
    )]
)

    # Customize the pie chart layout
    fig_pie.update_layout(
        title="Revenue Percentage by Coffee Type",
        title_font=dict(size=18, family='Arial', color='black'),  # Use `font` attributes
)

    # Display the two plots side by side using Streamlit columns
    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(fig_bar)

    with col2:
        st.plotly_chart(fig_pie)

    


    sales_by_coffee = df.groupby('coffee_name')['money'].sum().reset_index()
    fig3= px.bar(sales_by_coffee, x='coffee_name', y='money', title='Total Sales by Coffee Type', color='money' , height=400)
    # Update layout to make x and y labels bold and ensure all ticks are shown
    fig3.update_layout(
        width=1200,
        height=550,
        xaxis_title="<b>Coffee</b>",
        yaxis_title="<b>Money</b>",
        
)
    
    st.plotly_chart(fig3)

    # Group by coffee_name and week_day_name, counting the number of sales
    weekly_sales = df.groupby(['coffee_name', 'week_day_name'])['date'].count().reset_index().rename(columns={'date': 'count'})

    # Optional: Reorder the days for better visualization
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekly_sales['day'] = pd.Categorical(weekly_sales['week_day_name'], categories=day_order, ordered=True)
    weekly_sales = weekly_sales.sort_values(['coffee_name', 'day'])

    # Streamlit App
    st.title("Weekly Coffee Sales Visualization")

    # Add a selectbox for selecting days (multi-selection option)
    selected_days = st.multiselect("Select the Days of the Week to Display", options=day_order, default=day_order)

    # Filter the weekly sales data based on the selected days
    if selected_days:
           filtered_sales = weekly_sales[weekly_sales['week_day_name'].isin(selected_days)]
    else:
           filtered_sales = weekly_sales

    # Plotting the barplot using Seaborn in Streamlit
    plt.figure(figsize=(10, 5))
    sns.barplot(data=filtered_sales, x='day', y='count', hue='coffee_name', palette='coolwarm')

    plt.title('Weekly Coffee Sales by Coffee Name', fontsize=16)
    plt.xlabel('Day of the Week', fontsize=14)
    plt.ylabel('Number of Sales', fontsize=14)
    plt.xticks(rotation=45)
    plt.legend(title='Coffee Name')
    plt.grid(axis='y')
    plt.tight_layout()

    # Display the plot in Streamlit
    st.pyplot(plt)
   
    df_copy = df.copy()
    
    
    hourly_sales = df_copy.groupby(["hour"]).count()["date"].reset_index().rename(columns={"date":"count"})
    print(hourly_sales)
    # Assuming 'hourly_sales' is your DataFrame
    fig = px.bar(hourly_sales,
             x="hour",
             y="count",
             color="hour",
             title="<b>Coffee Hourly Sales by Date</b>",
             height=500)

    # Update layout to make x and y labels bold and ensure all ticks are shown
    fig.update_layout(
        width=1200,
        height=600,
        xaxis_title="<b>Hour</b>",
        yaxis_title="<b>Count</b>",
        title_x=0.5,  # Center the title 
        xaxis=dict(
             tickmode='linear',  # Ensure all x-axis values are shown
             tick0=0,            # Start at 0
             dtick=1             # Show every tick (step of 1)
    )
)

    # Display the chart
    st.plotly_chart(fig)    
    hourly_product_sales = df_copy.groupby(["hour" , "coffee_name"]).count()["date"].reset_index().rename(columns={"date":"count"})
    hourly_product_sales=hourly_product_sales.pivot(index='hour',columns='coffee_name',values='count').fillna(0).reset_index()

   

    # Define a color palette (you can use Plotly's built-in palettes or customize your own)
    color_palette = px.colors.qualitative.Pastel  # Using a pastel color palette for better visuals

    # Create subplot grid: 2 rows, 4 columns
    fig = make_subplots(rows=2, cols=4, subplot_titles=hourly_product_sales.columns[1:])

    # Initialize row and column counters
    row, col = 1, 1

    # Iterate over each coffee product
    for i, product in enumerate(hourly_product_sales.columns[1:], start=1):
       # Add a bar trace for each product with color palette
       fig.add_trace(
            go.Bar(x=hourly_product_sales['hour'], 
                 y=hourly_product_sales[product], 
                 name=product,
                 marker_color=color_palette[i % len(color_palette)]),  # Cycle through the palette
                 row=row, col=col
    )
    
        # Move to the next column, and next row if needed
       col += 1
       if col > 4:
          col = 1
          row += 1


    fig.update_layout(
         width=1400,
         height=700,  # Adjust height as needed
         title_text="<b>Hourly Sales by Product</b>",  # Title for the entire figure
         title_font=dict(size=20, family='Arial', color='darkblue'),  # Custom title font
         showlegend=False,  # Disable legend to avoid clutter
         plot_bgcolor='rgba(0, 0, 0, 0)',  # Transparent background for a clean look
         paper_bgcolor='rgba(255, 255, 255, 1)',  # White background
         font=dict(family="Arial", size=12, color="black")  # Global font style
)

    # Customize x and y axes
    fig.update_xaxes(title_text="<b>Hour</b>", showgrid=True, gridcolor='lightgray')
    fig.update_yaxes(title_text="<b>Sales Count</b>", showgrid=True, gridcolor='lightgray')

    # Show the plot
    st.plotly_chart(fig)

    

    # Title of the app
    st.title("Coffee Sales Analysis Project Report")

    # Input box for user to add text for the project report (editable)
    project_report = st.text_area(
    "Project Report Details Here: It will create Report",
    """This project analyzes the distribution of coffee sales across different types of coffee. 

We observe the frequency of sales and compare results across multiple factors such as daily, weekly, and monthly patterns. Custom visualizations are provided for better insights.

## Key Findings

* **Top Selling Products:**
  - Americano with Milk, Latte, and Cappuccino are the most popular choices, exhibiting significantly higher sales.
  
* **Underperforming Products:**
  - Hot Chocolate, Espresso, and Cocoa show considerably lower sales figures, indicating a need for improvement.
  
* **Profitability:**
  - The most profitable products include Americano with Milk, Latte, Cappuccino, and Americano, as their sales outpace those of other offerings.

* **Payment Methods:**
  - Over 90% of transactions are made via card, while only 7% are completed using cash.

## Additional Observations

- **Sales Patterns:**
  - Americano has the highest sales on **Mondays**, and Latte peaks on **Thursdays**.
  - Americano with Milk does particularly well on **Tuesdays**.
  - Cocoa and Espresso need advertising to boost their low sales figures.
  - Overall, coffee sales are higher on **weekends**.

## Sales Observations

- **Americano** sells best on **Mondays**, while **Latte** is the favorite on **Thursdays**.
- **Americano with Milk** does really well on **Tuesdays**.
- On the other hand, **Cocoa** and **Espresso** aren't selling much at all, so they could really use some advertising to attract more customers.
- **Cortado** is also not doing well in sales.
- Overall, we see that coffee sales are generally higher on the **weekends**.

# **Observations:**

1. As illustrated in the line chart above, Americano with Milk, Latte, and Cappuccino are the top-selling coffee types, whereas Cocoa and Espresso exhibit the lowest sales figures. 

2. Furthermore, Americano with Milk and Latte demonstrate an upward trend in their sales.

3. Notably, sales tend to rise significantly after May, particularly as we approach the colder months of June and July, which are more favorable for coffee consumption. Additionally, Americano with Milk and Latte demonstrate an upward trend in their sales.

# **Hourly Traffic Analysis for Coffee Products**

The visualizations above display the customer traffic for various coffee products over the course of a day. One clear pattern is that traffic peaks for all products around **10:00 AM**, with the spike being most evident for **Latte**.

During the evening, particularly between **6:00 PM and 8:00 PM**, we observe a preference shift toward beverages like **Cappuccino, Cocoa**, and **Hot Chocolate**, which see higher sales during these hours.

## **Key Insights**

This analysis provides valuable insights into daily and weekly customer behaviors. It highlights the top-selling coffee items and tracks how their popularity fluctuates throughout the day. 

These findings can support decisions in:
- **Inventory management** to avoid stockouts.
- Optimizing **vending machine layouts** for higher engagement.
- Pinpointing the best **restocking times** to meet demand more effectively.
"""
)

    # Display the project report
    if project_report:
        st.markdown(project_report)
    # Footer with "Created by Rajat Singh at Unified Mentor"
    st.markdown("***")  # Add a horizontal line for separation
    st.markdown(
       """
       <div style="text-align: center;">
           Created by <strong>Rajat Singh</strong> at <strong>Unified Mentor</strong>
       </div>
        """, 
    unsafe_allow_html=True
)


    
