import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Set page config
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# Load trained model
@st.cache_resource
def load_model():
    with open("house_price_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

# Load dataset and create encoder
@st.cache_data
def load_encoder():
    df = pd.read_csv("MagicBricks.csv")
    
    # SAME locality grouping used in notebook
    def grp_local(locality):
        locality = locality.lower()
        
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
    df['Locality'] = df['Locality'].apply(grp_local)
    
    # Create encoder
    locality_encoder = LabelEncoder()
    df['Locality'] = locality_encoder.fit_transform(df['Locality'])
    
    return locality_encoder, locality_encoder.classes_

# Load model and encoder
try:
    model = load_model()
    locality_encoder, localities = load_encoder()
except Exception as e:
    st.error(f"Error loading model or data: {str(e)}")
    st.stop()

# Title and description
st.title("🏠 House Price Predictor")
st.markdown("### Predict the price of a property based on its features")
st.markdown("---")

# Create two columns for input
col1, col2 = st.columns(2)

with col1:
    st.subheader("Property Details")
    
    # Numerical inputs
    area = st.number_input("Area (in sq ft)", 
                          min_value=100.0, 
                          max_value=10000.0, 
                          value=1000.0,
                          step=50.0,
                          help="Total area of the property in square feet")
    
    bhk = st.number_input("Number of BHK", 
                         min_value=1, 
                         max_value=10, 
                         value=2,
                         step=1,
                         help="Number of bedrooms, hall, kitchen")
    
    bathroom = st.number_input("Number of Bathrooms", 
                              min_value=1, 
                              max_value=10, 
                              value=2,
                              step=1)
    
    parking = st.number_input("Parking Spaces", 
                             min_value=0, 
                             max_value=5, 
                             value=1,
                             step=1)
    
    per_sqft = st.number_input("Price per sq ft (in ₹)", 
                              min_value=1000.0, 
                              max_value=50000.0, 
                              value=10000.0,
                              step=500.0,
                              help="Expected price per square foot")

with col2:
    st.subheader("Property Features")
    
    # Categorical inputs
    locality = st.selectbox("Locality", 
                           options=localities,
                           help="Select the locality of the property")
    
    furnishing = st.selectbox("Furnishing Status",
                             options=["Unfurnished", "Semi-Furnished", "Furnished"],
                             help="Furnishing level of the property")
    
    status = st.selectbox("Property Status",
                         options=["Ready_to_move", "Almost_ready"],
                         help="Construction status")
    
    transaction = st.selectbox("Transaction Type",
                              options=["New_Property", "Resale"],
                              help="Type of transaction")
    
    property_type = st.selectbox("Property Type",
                                options=["Apartment", "Builder_Floor"],
                                help="Type of property")

# Add some spacing
st.markdown("---")

# Predict button
if st.button("🔮 Predict Price", type="primary", use_container_width=True):
    try:
        # Encoding categorical inputs
        furnishing_map = {
            "Unfurnished": 0,
            "Semi-Furnished": 1,
            "Furnished": 2
        }
        
        status_map = {
            "Ready_to_move": 1,
            "Almost_ready": 0
        }
        
        transaction_map = {
            "New_Property": 1,
            "Resale": 0
        }
        
        type_map = {
            "Apartment": 0,
            "Builder_Floor": 1
        }
        
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
        
        # Convert crores to rupees
        price_rupees = prediction * 10000000
        
        # Display result
        st.markdown("---")
        st.subheader("💰 Predicted Price")
        
        # Create three columns for different price formats
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            st.metric("In Rupees", f"₹{price_rupees:,.2f}")
        
        with col_b:
            st.metric("In Lakhs", f"₹{price_rupees/100000:,.2f} L")
        
        with col_c:
            st.metric("In Crores", f"₹{price_rupees/10000000:,.2f} Cr")
        
        # Add a success message
        st.success("Prediction completed successfully!")
        
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        st.info("Please check if all inputs are valid and try again.")

# Add sidebar with information
with st.sidebar:
    st.header("📊 About")
    st.markdown("""
    This house price predictor uses machine learning to estimate property prices based on various features.
    
    **Features considered:**
    - Area (sq ft)
    - Number of BHK
    - Number of Bathrooms
    - Furnishing Status
    - Locality
    - Parking Spaces
    - Property Status
    - Transaction Type
    - Property Type
    - Price per sq ft
    
    **Note:** The prediction is an estimate based on the model trained on available data.
    """)
    
    st.header("💡 Tips")
    st.markdown("""
    - Enter accurate property details for better predictions
    - Price per sq ft should be based on recent market rates in the locality
    - The model works best for properties in Delhi NCR region
    """)
    
    st.header("📝 Instructions")
    st.markdown("""
    1. Fill in all property details
    2. Click "Predict Price" button
    3. View the estimated price in multiple formats
    """)

# Footer
st.markdown("---")
st.markdown("🏠 House Price Predictor | Made with ❤️")
