import streamlit as st
import pickle
import bz2
import numpy as np
from sklearn.preprocessing import StandardScaler
from streamlit_option_menu import option_menu

# Load Classification and Regression models
pickle_in = bz2.BZ2File('model/classification.pkl', 'rb')
R_pickle_in = bz2.BZ2File('model/regression.pkl', 'rb')
model_C = pickle.load(pickle_in)
model_R = pickle.load(R_pickle_in)

# Standardization
scaler = StandardScaler()

def main():
    st.set_page_config(page_title="Forest Fire Prediction", layout="wide")
    
    # st.sidebar.title("Navigation")
    # page = st.sidebar.radio("Go to", ["Home", "Fire Prediction"])
    with st.sidebar:
       page = option_menu(
        menu_title="Main Menu",
        options=["Home", "Fire Prediction"],
        icons=["house", "robot"],
        menu_icon="cast",
        default_index=0,
    )

    
    if page == "Home":
        home_page()
    elif page == "Fire Prediction":
        prediction_page()

def home_page():
    st.title("🔥 Forest Fire Prediction System")
    st.markdown("""
        Welcome to the **Algerian Forest Fire Prediction System**. This tool helps predict forest fire risks 
        using machine learning models.
        
        ### Features:
        - **Classification Model**: Predicts whether a forest is safe or in danger.
        - **Regression Model**: Provides a Fuel Moisture Code (FMC) index indicating fire hazard levels.
        
        Use the **Fire Prediction** page to input environmental data and get predictions.
    
    """)
    st.image("banner.jpg", use_container_width=True)
    

def prediction_page():
    st.title("🔥 Fire Prediction Tool")
    st.markdown("Input the required details to predict forest fire risks.")
    
    temperature = st.number_input("Temperature (°C)", min_value=-10.0, max_value=50.0, step=0.1)
    wind_speed = st.number_input("Wind Speed (km/h)", min_value=0.0, max_value=100.0, step=0.1)
    ffmc = st.number_input("Fuel Moisture Code (FFMC)", min_value=0.0, max_value=100.0, step=0.1)
    dmc = st.number_input("Duff Moisture Code (DMC)", min_value=0.0, max_value=200.0, step=0.1)
    isi = st.number_input("Initial Spread Index (ISI)", min_value=0.0, max_value=50.0, step=0.1)
    
    if st.button("Predict Fire Risk"):
        input_data = np.array([[temperature, wind_speed, ffmc, dmc, isi]])
        scaled_data = scaler.fit_transform(input_data)
        prediction = model_C.predict(scaled_data)[0]
        
        result_text = "Forest is in Danger! 🚨" if prediction == 1 else "Forest is Safe! ✅"
        st.subheader(result_text)
    
    if st.button("Predict Fire Weather Index"):
        input_data = np.array([[temperature, wind_speed, ffmc, dmc, isi]])
        scaled_data = scaler.fit_transform(input_data)
        output = model_R.predict(scaled_data)[0]
        
        if output > 15:
            st.subheader(f"FMC Index: {output:.4f} - High hazard warning! ⚠️")
        else:
            st.subheader(f"FMC Index: {output:.4f} - Low hazard, Safe. ✅")
    
if __name__ == "__main__":
    main()
