# House Price Prediction App - Student Version
# Created by: Shivansh Mishra (2nd Year B.Tech CSE)
# BBD University - Section 2A

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="🏠 House Price Predictor - Student Project",
    page_icon="🏠",
    layout="wide"
)

# Custom CSS for student-friendly styling
st.markdown("""
<style>
    /* Global Styles */
    body, .stMarkdown {
        color: #1A1A1A;
        background-color: #ffffff;
    }
    
    /* Main Header Styling */
    .main-header {
        font-size: 2.8rem;
        background: linear-gradient(120deg, #2E86AB, #4B59F7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Student Info Box */
    .student-info {
        background: linear-gradient(135deg, #6B48FF 0%, #8F6AFF 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(107, 72, 255, 0.2);
        transition: transform 0.3s ease;
    }
    .student-info:hover {
        transform: translateY(-5px);
    }
    
    /* Prediction Box */
    .prediction-box {
        background: linear-gradient(to right, #ffffff, #f8f9fa);
        padding: 1.8rem;
        border-radius: 12px;
        border-left: 5px solid #4B59F7;
        margin: 1rem 0;
        color: #1A1A1A;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
    }
    .prediction-box:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.1);
    }
    
    /* Metric Boxes */
    .metric-box {
        background: linear-gradient(135deg, #ffffff, #f0f7ff);
        padding: 1.2rem;
        border-radius: 12px;
        text-align: center;
        margin: 0.5rem 0;
        color: #1A1A1A;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
    }
    .metric-box:hover {
        transform: scale(1.02);
        background: linear-gradient(135deg, #f0f7ff, #e1edff);
    }
    
    /* Interactive Elements */
    button {
        transition: all 0.3s ease !important;
    }
    button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15) !important;
    }
    
    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 0.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #1A1A1A;
        background-color: transparent;
        transition: all 0.3s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #4B59F7;
        background-color: rgba(75, 89, 247, 0.1);
    }
    
    /* Chart Containers */
    .element-container {
        background: white;
        border-radius: 15px;
        padding: 1rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    .element-container:hover {
        box-shadow: 0 6px 20px rgba(0,0,0,0.1);
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background-color: #f8f9fa;
        border-right: 1px solid #e9ecef;
    }

    /* Style sidebar labels and headings */
    .st-emotion-cache-16idsys label,
    .st-emotion-cache-16idsys h2,
    .st-emotion-cache-16idsys h3,
    .st-emotion-cache-16idsys p,
    [data-testid="stSidebar"] label,
    .stMarkdown {
        color: black !important;
    }
    
    /* Style sidebar select boxes */
    .st-emotion-cache-16idsys .stSelectbox [data-baseweb=select],
    .st-emotion-cache-16idsys .stSelectbox div[data-baseweb=select] span {
        background-color: white !important;
    }
    
    /* Style sidebar number inputs */
    .st-emotion-cache-16idsys .stNumberInput [data-baseweb=input],
    .st-emotion-cache-16idsys .stNumberInput input {
        background-color: white !important;
    }

    /* Style input fields */
    [data-baseweb="input"],
    [data-baseweb="select"] {
        background-color: white !important;
    }

    /* Style select dropdown */
    [data-baseweb="popover"] {
        background-color: white !important;
    }
    [data-baseweb="select"] [data-testid="stMarkdown"] {
        background-color: white !important;
    }

    /* Ensure input text remains visible */
    input, select, .stSelectbox div[role="listbox"] {
        background-color: white !important;
        color: #2c3338 !important;
    }

    /* Style sidebar heading */
    .st-emotion-cache-16idsys h2,
    .st-emotion-cache-16idsys h3 {
        font-weight: 600;
    }
    
    /* Footer Styling */
    .footer {
        background: linear-gradient(135deg, #f8f9fa, #ffffff);
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        margin-top: 2rem;
        transition: all 0.3s ease;
    }
    .footer:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Title and student information
st.markdown('<h1 class="main-header">🏠 House Price Prediction System</h1>', unsafe_allow_html=True)

st.markdown("""
<div class="student-info">
    <h3>📚 Student Project</h3>
    <p><strong>Created by:</strong> Shivansh Mishra</p>
    <p><strong>University:</strong> BBD University | <strong>Course:</strong> B.Tech CSE (CC & ML)</p>
    <p><strong>Year:</strong> 2nd Year | <strong>Section:</strong> 2A</p>
    <p><em>"Learning by building real-world applications!"</em></p>
</div>
""", unsafe_allow_html=True)

# Load and cache data
@st.cache_data
def load_data():
    """Load the house price dataset"""
    try:
        df = pd.read_csv('house_prices_8000.csv')
        return df
    except FileNotFoundError:
        st.error("❌ Dataset file not found! Please make sure 'house_prices_8000.csv' is in the same directory.")
        return None

# Train model
@st.cache_resource
def train_models(df):
    """Train machine learning models"""
    # Prepare features
    features_to_drop = ['id', 'date_sold', 'price', 'price_per_sqft']
    X = df.drop(features_to_drop, axis=1)
    
    # Handle categorical variables (simple approach for students)
    X_encoded = pd.get_dummies(X, columns=['city', 'neighborhood'], drop_first=True)
    y = df['price']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
    
    # Train two models (keeping it simple for 2nd year level)
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=50, random_state=42)  # Reduced for faster training
    }
    
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        results[name] = {
            'model': model,
            'r2': r2_score(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred)
        }
    
    return results, X_encoded.columns.tolist(), X_test, y_test

def main():
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Show basic dataset info
    st.markdown("## 📊 Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-box">
            <h3>{len(df):,}</h3>
            <p>Total Properties</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-box">
            <h3>{df['city'].nunique()}</h3>
            <p>Cities Covered</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-box">
            <h3>₹{df['price'].mean()/100000:.1f}L</h3>
            <p>Average Price</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-box">
            <h3>{df.shape[1]}</h3>
            <p>Features</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Train models
    with st.spinner('🤖 Training machine learning models...'):
        model_results, feature_names, X_test, y_test = train_models(df)
    
    # Model performance
    st.markdown("## 🎯 Model Performance")
    col1, col2 = st.columns(2)
    
    for i, (name, results) in enumerate(model_results.items()):
        col = col1 if i == 0 else col2
        with col:
            st.markdown(f"""
            <div class="prediction-box">
                <h4>{name}</h4>
                <p><strong>Accuracy (R²):</strong> {results['r2']:.3f}</p>
                <p><strong>Error (RMSE):</strong> ₹{results['rmse']/100000:.2f} Lakhs</p>
                <p><strong>Mean Error:</strong> ₹{results['mae']/100000:.2f} Lakhs</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Best model
    best_model_name = max(model_results.keys(), key=lambda x: model_results[x]['r2'])
    best_model = model_results[best_model_name]['model']
    
    st.success(f"🏆 Best Model: {best_model_name} with {model_results[best_model_name]['r2']:.1%} accuracy!")
    
    # Sidebar for prediction
    st.sidebar.markdown("""
    <div style='background: linear-gradient(135deg, #6B48FF 0%, #8F6AFF 100%); padding: 15px; border-radius: 10px; margin-bottom: 20px;'>
        <h2 style='color: white; text-align: center; margin: 0;'>🏡 Predict House Price</h2>
        <p style='color: white; text-align: center; margin: 10px 0 0 0;'>Fill in the property details below:</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Location Details
    st.sidebar.markdown("### 📍 Location Details")
    city = st.sidebar.selectbox("🏙️ City", df['city'].unique(), 
                              help="Select the city where the property is located")
    neighborhood = st.sidebar.selectbox("🏘️ Neighborhood", df['neighborhood'].unique(), 
                                      help="Select the neighborhood within the chosen city")
    
    # Property Basics
    st.sidebar.markdown("### 🏠 Property Basics")
    col_a, col_b = st.sidebar.columns(2)
    with col_a:
        sqft_living = st.number_input("Living Area (sqft)", 500, 6000, 2000,
                                    help="Total living area in square feet")
        bedrooms = st.number_input("Bedrooms", 1, 10, 3,
                                 help="Number of bedrooms")
    
    with col_b:
        bathrooms = st.number_input("Bathrooms", 1.0, 5.5, 2.0, 0.5,
                                  help="Number of bathrooms (including half baths)")
        grade = st.number_input("Grade (1-13)", 1, 13, 7,
                              help="Overall grade given to the housing unit")
    
    # Additional Details
    st.sidebar.markdown("### 📊 Additional Details")
    sqft_lot = st.sidebar.number_input("Lot Size (sqft)", 1000, 35000, 8000,
                                     help="Total lot size in square feet")
    yr_built = st.sidebar.number_input("Year Built", 1900, 2024, 1990,
                                     help="Year the property was built")
    
    # Predict button
    if st.sidebar.button("🔮 Predict Price!", type="primary"):
        # Create input for prediction
        input_data = {
            'zipcode': df[df['city'] == city]['zipcode'].iloc[0],
            'lat': df[df['city'] == city]['lat'].mean(),
            'long': df[df['city'] == city]['long'].mean(),
            'sqft_living': sqft_living,
            'sqft_lot': sqft_lot,
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'floors': 2,  # Default value
            'waterfront': 0,  # Default value
            'view': 2,  # Default value
            'condition': 3,  # Default value
            'grade': grade,
            'sqft_above': int(sqft_living * 0.8),  # Estimate
            'sqft_basement': int(sqft_living * 0.2),  # Estimate
            'yr_built': yr_built,
            'yr_renovated': 0,  # Default value
            'garage': 1,  # Default value
            'parking': 2,  # Default value
            'hoa_monthly': 50  # Default value
        }
        
        # Add categorical encoding
        for col in feature_names:
            if col.startswith('city_') or col.startswith('neighborhood_'):
                input_data[col] = 0
        
        # Set selected city and neighborhood
        city_col = f'city_{city}'
        neighborhood_col = f'neighborhood_{neighborhood}'
        if city_col in feature_names:
            input_data[city_col] = 1
        if neighborhood_col in feature_names:
            input_data[neighborhood_col] = 1
        
        # Create DataFrame and predict
        input_df = pd.DataFrame([input_data])
        input_df = input_df.reindex(columns=feature_names, fill_value=0)
        
        prediction = best_model.predict(input_df)[0]
        
        # Display prediction
        st.markdown(f"""
        <div class="prediction-box">
            <h2>🎯 Predicted House Price</h2>
            <h1 style="color: #2E86AB;">₹{prediction/100000:.2f} Lakhs</h1>
            <p style="font-size: 1.1em;">≈ ₹{prediction:,.0f}</p>
            <p><strong>Model Used:</strong> {best_model_name}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Compare with city average
        city_avg = df[df['city'] == city]['price'].mean()
        if prediction > city_avg:
            st.info(f"💰 This property is ₹{(prediction-city_avg)/100000:.2f} lakhs above the {city} average!")
        else:
            st.info(f"💡 This property is ₹{(city_avg-prediction)/100000:.2f} lakhs below the {city} average!")
    
    # Data Visualization Section
    st.markdown("## 📈 Data Insights")
    
    # Create simple visualizations
    tab1, tab2, tab3 = st.tabs(["🏙️ City Prices", "📊 Price Distribution", "🔍 Dataset Sample"])
    
    with tab1:
        # City-wise average prices
        city_prices = df.groupby('city')['price'].mean().sort_values(ascending=False)
        fig = px.bar(
            x=city_prices.index,
            y=city_prices.values/100000,
            title="Average House Prices by City",
            labels={'x': 'City', 'y': 'Average Price (Lakhs)'},
            color=city_prices.values,
            color_continuous_scale='viridis'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Price distribution
        fig = px.histogram(
            df,
            x='price',
            nbins=30,
            title="Distribution of House Prices",
            labels={'price': 'Price (INR)', 'count': 'Number of Properties'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Show sample data
        st.markdown("### 🔍 Sample of Dataset")
        st.dataframe(df.head(10), use_container_width=True)
        
        st.markdown("### 📋 Dataset Features")
        st.write("**Location Features:** City, Neighborhood, Zipcode, Latitude, Longitude")
        st.write("**Property Features:** Living Area, Lot Size, Bedrooms, Bathrooms, Floors")
        st.write("**Quality Features:** Grade, Condition, View, Waterfront")
        st.write("**Additional:** Year Built, Garage, Parking, HOA Monthly")

    # Learning section
    st.markdown("---")
    st.markdown("## 🎓 What I Learned from This Project")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **📊 Data Science**
        - Loading and exploring datasets
        - Data cleaning and preprocessing
        - Statistical analysis and insights
        - Creating meaningful visualizations
        """)
    
    with col2:
        st.markdown("""
        **🤖 Machine Learning**
        - Training regression models
        - Comparing model performance
        - Understanding accuracy metrics
        - Making predictions on new data
        """)
    
    with col3:
        st.markdown("""
        **💻 Software Development**
        - Building web applications
        - User interface design
        - Code organization
        - Professional documentation
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div class="footer">
        <h4 style="color: #4B59F7;">👨‍🎓 Student Project - BBD University</h4>
        <p><strong style="color: #6B48FF;">Shivansh Mishra</strong> - 2nd Year B.Tech CSE (CC & ML) - Section 2A</p>
        <p style="color: #666; font-style: italic;">"Building the future, one project at a time!"</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()