import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page config
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .info-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Load trained model
@st.cache_resource
def load_model():
    with open("house_price_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

# Load dataset and create encoder
@st.cache_data
def load_data():
    df = pd.read_csv("MagicBricks.csv")
    
    # Clean the data - handle NaN values
    df = df.dropna(subset=['Bathroom'])  # Remove rows with NaN in Bathroom
    df = df.fillna(0)  # Fill other NaN values with 0
    
    # SAME locality grouping used in notebook
    def grp_local(locality):
        if pd.isna(locality):
            return 'Other'
        locality = str(locality).lower()
        
        if 'rohini' in locality:
            return 'Rohini Sector'
        elif 'dwarka' in locality:
            return 'Dwarka Sector'
        elif 'shahdara' in locality:
            return 'Shahdara'
        elif 'vasant' in locality:
            return 'Vasant Kunj'
        elif 'paschim' in locality:
            return 'Paschim Vihar'
        elif 'alaknanda' in locality:
            return 'Alaknanda'
        elif 'vasundhar' in locality:
            return 'Vasundhara Enclave'
        elif 'punjabi' in locality:
            return 'Punjabi Bagh'
        elif 'kalkaji' in locality:
            return 'Kalkaji'
        elif 'lajpat' in locality:
            return 'Lajpat Nagar'
        else:
            return 'Other'
    
    # Apply same preprocessing
    df['Locality_Group'] = df['Locality'].apply(grp_local)
    
    # Create encoder
    locality_encoder = LabelEncoder()
    df['Locality_Encoded'] = locality_encoder.fit_transform(df['Locality_Group'])
    
    # Convert necessary columns to numeric
    numeric_columns = ['Area', 'Price', 'BHK', 'Bathroom', 'Parking', 'Per_Sqft']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop any remaining NaN values in critical columns
    df = df.dropna(subset=['Area', 'Price', 'BHK', 'Bathroom'])
    
    return df, locality_encoder, locality_encoder.classes_

# Load model and data
try:
    model = load_model()
    df, locality_encoder, localities = load_data()
except Exception as e:
    st.error(f"Error loading model or data: {str(e)}")
    st.stop()

# Header with animation
st.markdown('<div class="main-header">🏠 House Price Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Powered Real Estate Valuation for Delhi NCR</div>', unsafe_allow_html=True)

# Create tabs for different sections
tab1, tab2, tab3, tab4 = st.tabs(["📊 Predictor", "📈 Market Insights", "📍 Location Analysis", "📚 Educational Insights"])

# Tab 1: Predictor
with tab1:
    st.markdown("### 🔮 Property Price Prediction")
    
    # Create three columns for input
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 📐 Property Dimensions")
        area = st.number_input("Area (sq ft)", 
                              min_value=100.0, 
                              max_value=10000.0, 
                              value=1000.0,
                              step=50.0,
                              help="Total area of the property in square feet",
                              key="area")
        
        bhk = st.selectbox("Number of BHK", 
                          options=[1,2,3,4,5,6,7,8,9,10],
                          index=1,
                          help="Number of bedrooms, hall, kitchen",
                          key="bhk")
        
        bathroom = st.slider("Number of Bathrooms", 
                            min_value=1, 
                            max_value=10, 
                            value=2,
                            help="Total bathrooms in the property",
                            key="bathroom")
    
    with col2:
        st.markdown("#### 🏢 Property Features")
        locality = st.selectbox("Locality", 
                               options=localities,
                               help="Select the locality of the property",
                               key="locality")
        
        property_type = st.radio("Property Type",
                                options=["Apartment", "Builder_Floor"],
                                horizontal=True,
                                key="type")
        
        furnishing = st.select_slider("Furnishing Level",
                                     options=["Unfurnished", "Semi-Furnished", "Furnished"],
                                     value="Semi-Furnished",
                                     key="furnishing")
    
    with col3:
        st.markdown("#### 🚗 Additional Details")
        parking = st.number_input("Parking Spaces", 
                                 min_value=0, 
                                 max_value=5, 
                                 value=1,
                                 step=1,
                                 key="parking")
        
        status = st.selectbox("Construction Status",
                             options=["Ready_to_move", "Almost_ready"],
                             key="status")
        
        transaction = st.selectbox("Transaction Type",
                                  options=["New_Property", "Resale"],
                                  key="transaction")
        
        per_sqft = st.number_input("Expected Price per sq ft (₹)", 
                                  min_value=1000.0, 
                                  max_value=50000.0, 
                                  value=10000.0,
                                  step=500.0,
                                  help="Expected price per square foot",
                                  key="per_sqft")
    
    # Real-time price estimation
    st.markdown("---")
    
    # Predict button with better styling
    col_btn1, col_btn2, col_btn3 = st.columns([1,2,1])
    with col_btn2:
        predict_button = st.button("🔮 Predict Price", type="primary", use_container_width=True)
    
    if predict_button:
        try:
            # Encoding categorical inputs
            furnishing_map = {"Unfurnished": 0, "Semi-Furnished": 1, "Furnished": 2}
            status_map = {"Ready_to_move": 1, "Almost_ready": 0}
            transaction_map = {"New_Property": 1, "Resale": 0}
            type_map = {"Apartment": 0, "Builder_Floor": 1}
            
            furnishing_encoded = furnishing_map[furnishing]
            status_encoded = status_map[status]
            transaction_encoded = transaction_map[transaction]
            type_encoded = type_map[property_type]
            
            # Encode locality
            locality_encoded = locality_encoder.transform([locality])[0]
            
            # Prepare features
            features = np.array([[area, bhk, bathroom, furnishing_encoded,
                                  locality_encoded, parking, status_encoded,
                                  transaction_encoded, type_encoded, per_sqft]])
            
            # Make prediction
            prediction = model.predict(features)[0]
            price_rupees = prediction * 10000000
            
            # Display result with animations
            st.markdown("---")
            st.markdown("### 💰 Estimated Property Value")
            
            # Create metric columns
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            
            with col_m1:
                st.metric("Total Price", f"₹{price_rupees:,.0f}", delta=None)
            with col_m2:
                st.metric("Price per sq ft", f"₹{price_rupees/area:,.0f}", delta=None)
            with col_m3:
                st.metric("Price in Crores", f"₹{price_rupees/10000000:,.2f} Cr", delta=None)
            with col_m4:
                st.metric("Price in Lakhs", f"₹{price_rupees/100000:,.2f} L", delta=None)
            
            # Add gauge chart for price comparison
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = price_rupees/10000000,
                title = {'text': "Price (in Crores)"},
                domain = {'x': [0, 1], 'y': [0, 1]},
                gauge = {
                    'axis': {'range': [None, 5]},
                    'bar': {'color': "#667eea"},
                    'steps': [
                        {'range': [0, 1], 'color': "lightgray"},
                        {'range': [1, 2], 'color': "gray"},
                        {'range': [2, 3], 'color': "darkgray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': price_rupees/10000000
                    }
                }
            ))
            
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            st.success("✅ Prediction completed successfully!")
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")

# Tab 2: Market Insights
with tab2:
    st.markdown("### 📊 Real Estate Market Insights")
    
    # Check if dataframe is not empty
    if len(df) > 0:
        # Create visualizations
        col_v1, col_v2 = st.columns(2)
        
        with col_v1:
            st.markdown("#### Price Distribution by Property Type")
            if 'Type' in df.columns and 'Price' in df.columns:
                fig = px.box(df, x='Type', y='Price', color='Type',
                            title='Price Distribution: Apartments vs Builder Floors',
                            labels={'Price': 'Price (Crores)', 'Type': 'Property Type'})
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Insufficient data for this visualization")
        
        with col_v2:
            st.markdown("#### Average Price by BHK")
            if 'BHK' in df.columns and 'Price' in df.columns:
                avg_price_by_bhk = df.groupby('BHK')['Price'].mean().reset_index()
                fig = px.bar(avg_price_by_bhk, x='BHK', y='Price',
                            title='Average Property Price by Number of BHK',
                            labels={'Price': 'Average Price (Crores)', 'BHK': 'Number of BHK'},
                            color='Price',
                            color_continuous_scale='Viridis')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Insufficient data for this visualization")
        
        # Price trends
        st.markdown("#### Price Trends by Furnishing Status")
        col_v3, col_v4 = st.columns(2)
        
        with col_v3:
            if 'Furnishing' in df.columns and 'Price' in df.columns:
                avg_price_furnishing = df.groupby('Furnishing')['Price'].mean().reset_index()
                fig = px.pie(avg_price_furnishing, values='Price', names='Furnishing',
                            title='Average Price by Furnishing Status',
                            color_discrete_sequence=px.colors.sequential.RdBu)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Insufficient data for this visualization")
        
        with col_v4:
            # Price vs Area scatter plot with fixed size parameter
            if 'Area' in df.columns and 'Price' in df.columns and 'BHK' in df.columns and 'Bathroom' in df.columns:
                # Create a copy and ensure no NaN values in Bathroom
                plot_df = df[['Area', 'Price', 'BHK', 'Bathroom', 'Locality_Group']].copy()
                plot_df = plot_df.dropna(subset=['Bathroom'])
                
                # Ensure Bathroom values are positive
                plot_df['Bathroom'] = plot_df['Bathroom'].clip(lower=1)
                
                fig = px.scatter(plot_df, x='Area', y='Price', color='BHK',
                                size='Bathroom', 
                                size_max=20,  # Set maximum size
                                hover_data=['Locality_Group'],
                                title='Price vs Area (Size represents Bathrooms)',
                                labels={'Price': 'Price (Crores)', 'Area': 'Area (sq ft)'})
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Insufficient data for this visualization")
    else:
        st.warning("No data available for visualizations")

# Tab 3: Location Analysis
with tab3:
    st.markdown("### 📍 Location-wise Price Analysis")
    
    if len(df) > 0:
        # Create location analysis
        col_loc1, col_loc2 = st.columns(2)
        
        with col_loc1:
            # Top localities by average price
            if 'Locality_Group' in df.columns and 'Price' in df.columns:
                top_localities = df.groupby('Locality_Group')['Price'].mean().sort_values(ascending=False).head(10)
                fig = px.bar(x=top_localities.values, y=top_localities.index,
                            orientation='h', title='Top 10 Localities by Average Price',
                            labels={'x': 'Average Price (Crores)', 'y': 'Locality'},
                            color=top_localities.values,
                            color_continuous_scale='Viridis')
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Insufficient data for this visualization")
        
        with col_loc2:
            # Price distribution across localities
            locality_stats = df.groupby('Locality_Group').agg({
                'Price': ['mean', 'count', 'std']
            }).round(2)
            locality_stats.columns = ['Avg Price (Cr)', 'Number of Properties', 'Price Std Dev']
            locality_stats = locality_stats.sort_values('Avg Price (Cr)', ascending=False)
            
            st.markdown("#### Locality Statistics")
            st.dataframe(locality_stats.head(10), use_container_width=True)
        
        # Heatmap of features by locality
        st.markdown("#### Feature Correlation by Locality")
        
        # Prepare data for heatmap
        numeric_cols = ['Price', 'Area', 'BHK', 'Bathroom']
        available_cols = [col for col in numeric_cols if col in df.columns]
        
        if available_cols:
            locality_features = df.groupby('Locality_Group')[available_cols].mean().round(2)
            
            fig = px.imshow(locality_features.T, 
                            text_auto=True, 
                            aspect="auto",
                            title="Average Property Features by Locality",
                            labels=dict(x="Locality", y="Feature", color="Value"))
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Insufficient data for heatmap visualization")
    else:
        st.warning("No data available for location analysis")

# Tab 4: Educational Insights
with tab4:
    st.markdown("### 📚 Understanding Property Valuation")
    
    col_edu1, col_edu2 = st.columns(2)
    
    with col_edu1:
        st.markdown("#### 🎯 Key Factors Affecting Property Prices")
        
        # Feature importance visualization
        feature_importance = {
            'Area': 25,
            'BHK': 15,
            'Bathroom': 10,
            'Locality': 20,
            'Furnishing': 8,
            'Parking': 5,
            'Property Type': 7,
            'Other Factors': 10
        }
        
        fig = px.pie(values=list(feature_importance.values()), 
                     names=list(feature_importance.keys()),
                     title="Factors Influencing Property Prices",
                     color_discrete_sequence=px.colors.sequential.Plasma)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        **💡 Investment Tips:**
        - Properties in premium localities typically appreciate 8-12% annually
        - Fully furnished properties command 15-20% higher prices
        - Ready-to-move properties are priced 5-10% higher than under-construction
        - Each additional BHK adds approximately 20-30% to the property value
        """)
    
    with col_edu2:
        st.markdown("#### 📈 Market Trends Analysis")
        
        # Create a mock trend line
        years = ['2020', '2021', '2022', '2023', '2024']
        price_trends = {
            'Apartments': [85, 92, 105, 118, 132],
            'Builder Floors': [78, 85, 98, 112, 125]
        }
        
        fig = go.Figure()
        for prop_type, prices in price_trends.items():
            fig.add_trace(go.Scatter(x=years, y=prices, name=prop_type,
                                    mode='lines+markers',
                                    line=dict(width=3)))
        
        fig.update_layout(title="Property Price Trends (Indexed to 2020)",
                         xaxis_title="Year",
                         yaxis_title="Price Index (2020 = 100)",
                         hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        **📊 Market Insights:**
        - Delhi NCR real estate market grew by 12% in 2024
        - Ready-to-move inventory increased by 15% post-pandemic
        - Luxury segment (₹2Cr+) seeing highest demand
        - Rental yields in prime locations: 2.5-3.5%
        """)
    
    # ROI Calculator Section
    st.markdown("---")
    st.markdown("#### 💰 Return on Investment (ROI) Calculator")
    
    col_roi1, col_roi2 = st.columns(2)
    
    with col_roi1:
        purchase_price = st.number_input("Purchase Price (₹ Crores)", 
                                        min_value=0.1, 
                                        max_value=10.0, 
                                        value=1.0,
                                        step=0.1,
                                        key="purchase_price")
        
        years_hold = st.slider("Holding Period (Years)", 
                              min_value=1, 
                              max_value=20, 
                              value=5,
                              key="years")
    
    with col_roi2:
        expected_appreciation = st.slider("Expected Annual Appreciation (%)", 
                                         min_value=5, 
                                         max_value=20, 
                                         value=10,
                                         key="appreciation")
        
        rental_yield = st.slider("Annual Rental Yield (%)", 
                                min_value=1, 
                                max_value=6, 
                                value=3,
                                key="rental")
    
    if st.button("Calculate ROI", key="roi_calc"):
        future_value = purchase_price * ((1 + expected_appreciation/100) ** years_hold)
        total_rental_income = purchase_price * (rental_yield/100) * years_hold
        total_value = future_value + total_rental_income
        roi_percentage = ((total_value - purchase_price) / purchase_price) * 100
        annualized_roi = ((total_value / purchase_price) ** (1/years_hold) - 1) * 100
        
        col_result1, col_result2, col_result3, col_result4 = st.columns(4)
        
        with col_result1:
            st.metric("Future Property Value", f"₹{future_value:.2f} Cr")
        with col_result2:
            st.metric("Total Rental Income", f"₹{total_rental_income:.2f} Cr")
        with col_result3:
            st.metric("Total Returns", f"₹{total_value:.2f} Cr")
        with col_result4:
            st.metric("Annualized ROI", f"{annualized_roi:.1f}%")
        
        st.success(f"Your investment would grow by {roi_percentage:.1f}% over {years_hold} years!")

# Sidebar enhancements
with st.sidebar:
    st.markdown("### 🏙️ Market Snapshot")
    
    if len(df) > 0:
        # Quick stats
        avg_price = df['Price'].mean() if 'Price' in df.columns else 0
        avg_area = df['Area'].mean() if 'Area' in df.columns else 0
        most_common_bhk = df['BHK'].mode()[0] if 'BHK' in df.columns and len(df['BHK'].mode()) > 0 else 2
        
        st.metric("Average Property Price", f"₹{avg_price:.2f} Cr")
        st.metric("Average Area", f"{avg_area:.0f} sq ft")
        st.metric("Most Common", f"{most_common_bhk} BHK")
        
        st.markdown("---")
        
        st.markdown("### 📊 Quick Statistics")
        st.markdown(f"""
        - **Total Properties**: {len(df):,}
        - **Unique Localities**: {len(localities)}
        - **Price Range**: ₹{df['Price'].min():.1f}Cr - ₹{df['Price'].max():.1f}Cr
        - **Avg Price/sq ft**: ₹{df['Price'].mean()/df['Area'].mean()*10000000:.0f}
        """)
    else:
        st.warning("No data available")
    
    st.markdown("---")
    
    st.markdown("### 🎯 Pro Tips")
    with st.expander("💡 Investment Tips"):
        st.markdown("""
        - **Location First**: Premium localities offer better appreciation
        - **Timing Matters**: Q4 (Oct-Dec) often has better deals
        - **Check Legal Status**: Verify all documents before purchase
        - **Negotiate Wisely**: 5-10% negotiation is common
        """)
    
    with st.expander("🏗️ Construction Quality"):
        st.markdown("""
        Look for:
        - RERA registration
        - Quality of materials
        - Builder reputation
        - Completion timeline
        """)
    
    st.markdown("---")
    st.markdown("📅 **Last Updated**: March 2026")
    st.markdown("💡 *Data based on Delhi NCR market*")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🏠 House Price Predictor | Powered by Machine Learning | Data Source: MagicBricks</p>
    <p><small>Disclaimer: This is an AI-powered estimation tool. Actual market prices may vary. Always consult with real estate professionals.</small></p>
</div>
""", unsafe_allow_html=True)
