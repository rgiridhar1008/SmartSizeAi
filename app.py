import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# =========================
# Page Config & Styling
# =========================
st.set_page_config(
    page_title="SmartSize AI 👕",
    page_icon="👕",
    layout="wide",
)

# Custom CSS for better design
st.markdown("""
    <style>
        body {
            background-color: #F8F9FA;
        }
        .title {
            text-align: center;
            color: #2E4053;
            font-family: 'Poppins', sans-serif;
            padding: 10px 0;
        }
        .stButton>button {
            background-color: #007BFF;
            color: white;
            font-weight: bold;
            border-radius: 10px;
            height: 45px;
            width: 100%;
        }
        .stButton>button:hover {
            background-color: #0056b3;
            color: white;
        }
        .result-box {
            background: linear-gradient(135deg, #007BFF, #00C9A7);
            border-radius: 12px;
            padding: 25px;
            text-align: center;
            color: white;
            font-family: 'Poppins', sans-serif;
            box-shadow: 0px 4px 15px rgba(0,0,0,0.2);
        }
        .footer {
            text-align: center;
            font-size: 13px;
            color: gray;
            padding-top: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# =========================
# Load Model & Preprocessing
# =========================
loaded_model = joblib.load('SMARTSIZE_MODEL.pkl')

dummy_data = {
    'Age': np.random.rand(100) * 50 + 18,
    'Height_cm': np.random.rand(100) * 50 + 150,
    'Weight_kg': np.random.rand(100) * 100 + 30,
    'Chest_cm': np.random.rand(100) * 30 + 70,
    'Waist_cm': np.random.rand(100) * 30 + 60,
    'Hip_cm': np.random.rand(100) * 30 + 80,
    'Gender': np.random.choice(['Female', 'Male'], 100),
    'Fit_Preference': np.random.choice(['Slim', 'Regular', 'Loose'], 100)
}
dummy_df_for_preprocessing = pd.DataFrame(dummy_data)

numeric_cols_input = ['Age', 'Height_cm', 'Weight_kg', 'Chest_cm', 'Waist_cm', 'Hip_cm']
categorical_cols_input = ['Gender', 'Fit_Preference']

scaler = StandardScaler()
scaler.fit(dummy_df_for_preprocessing[numeric_cols_input])

encoder = OneHotEncoder(sparse_output=False)
encoder.fit(dummy_df_for_preprocessing[categorical_cols_input])

trained_columns_order = scaler.get_feature_names_out(numeric_cols_input).tolist() + encoder.get_feature_names_out(categorical_cols_input).tolist()

# =========================
# Page Layout
# =========================
st.markdown("<h1 class='title'>👕 SmartSize AI -  Machine Learning Based Clothing Size Recommender </h1>", unsafe_allow_html=True)
st.markdown("#### Get your perfect fit instantly — powered by Machine Learning")

col1, col2 = st.columns([1.2, 1])

# =========================
# Input Section
# =========================
with col1:
    st.subheader("📏 Enter Your Measurements")

    unit_option = st.radio("Select Unit for Body Measurements:", ["Centimeters (cm)", "Feet & Inches (ft/in)"], horizontal=True)

    age = st.slider('Age', min_value=15, max_value=100, value=30)

    # Height input
    if unit_option == "Centimeters (cm)":
        height_cm = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=170.0)
    else:
        st.markdown("**Height (ft/in):**")
        col_h1, col_h2 = st.columns(2)
        with col_h1:
            height_feet = st.number_input("Feet", min_value=1, max_value=8, value=5)
        with col_h2:
            height_inches = st.number_input("Inches", min_value=0, max_value=11, value=7)
        # Convert ft+in → cm
        height_cm = (height_feet * 12 + height_inches) * 2.54

    # Other inputs
    weight = st.number_input('Weight (kg)', min_value=10.0, max_value=200.0, value=70.0)
    chest = st.number_input(f'Chest ({ "in" if unit_option == "Feet & Inches (ft/in)" else "cm"})', 
                            min_value=20.0, max_value=60.0 if unit_option == "Feet & Inches (ft/in)" else 150.0, 
                            value=36.0 if unit_option == "Feet & Inches (ft/in)" else 90.0)
    waist = st.number_input(f'Waist ({ "in" if unit_option == "Feet & Inches (ft/in)" else "cm"})', 
                            min_value=15.0, max_value=60.0 if unit_option == "Feet & Inches (ft/in)" else 150.0, 
                            value=30.0 if unit_option == "Feet & Inches (ft/in)" else 75.0)
    hip = st.number_input(f'Hip ({ "in" if unit_option == "Feet & Inches (ft/in)" else "cm"})', 
                          min_value=20.0, max_value=70.0 if unit_option == "Feet & Inches (ft/in)" else 160.0, 
                          value=40.0 if unit_option == "Feet & Inches (ft/in)" else 100.0)

    # Convert inches → cm if needed
    if unit_option == "Feet & Inches (ft/in)":
        chest *= 2.54
        waist *= 2.54
        hip *= 2.54

    st.subheader("👕 Your Preferences")
    gender = st.selectbox('Gender', ['Female', 'Male'])
    fit_preference = st.selectbox('Fit Preference', ['Slim', 'Regular', 'Loose'])

    # Prepare input
    user_input_df = pd.DataFrame({
        'Age': [age],
        'Height_cm': [height_cm],
        'Weight_kg': [weight],
        'Chest_cm': [chest],
        'Waist_cm': [waist],
        'Hip_cm': [hip],
        'Gender': [gender],
        'Fit_Preference': [fit_preference]
    })

    # Process user input
    scaled_numeric_input = scaler.transform(user_input_df[numeric_cols_input])
    scaled_numeric_input_df = pd.DataFrame(scaled_numeric_input, columns=numeric_cols_input)

    encoded_categorical_input = encoder.transform(user_input_df[categorical_cols_input])
    encoded_categorical_input_df = pd.DataFrame(encoded_categorical_input, columns=encoder.get_feature_names_out(categorical_cols_input))

    processed_user_input = pd.concat([scaled_numeric_input_df, encoded_categorical_input_df], axis=1)
    processed_user_input = processed_user_input.reindex(columns=trained_columns_order, fill_value=0)

    if st.button('🔍 Get Recommended Size'):
        recommended_size = loaded_model.predict(processed_user_input)
        st.markdown(f"""
            <div class='result-box'>
                <h2>🎯 Recommended Size: <span style='color:#FFFDD0;'>{recommended_size[0]}</span></h2>
                <p>Based on your measurements and fit preference.</p>
            </div>
        """, unsafe_allow_html=True)

# =========================
# Result / Info Section
# =========================
with col2:
    st.image("https://cdn-icons-png.flaticon.com/512/4149/4149678.png", width=250)
    st.info("💡 Tip: Try different fit preferences to see how the recommended size changes.")
    st.divider()
    st.markdown("### 🧠 How it works:")
    st.markdown("""
    1. Your measurements are normalized.
    2. Encoded with fit preference & gender.
    3. Model predicts the best-matching clothing size.
    """)
    st.divider()
    st.markdown("### 📦 Model Details:")
    st.markdown("""
    - Model: SVM (trained on 50,000+ records)  
    - Input features: Age, Height, Weight, Chest, Waist, Hip, Gender, Fit  
    - Output: Recommended clothing size (XS,S, M, L, XL,XXL)
    """)

st.markdown("<div class='footer'>© 2025 SmartSize AI | Powered by Machine Learning</div>", unsafe_allow_html=True)
