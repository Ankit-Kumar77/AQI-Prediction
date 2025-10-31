import streamlit as st
import joblib
import datetime
import pandas as pd

# Use caching to load the model only once, improving performance.
@st.cache_resource
def load_model(model_path='rf_model.pkl'):
    """
    Loads the pre-trained model from a file.
    Handles FileNotFoundError and other exceptions.
    """
    try:
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        st.error(f"Error: Model file not found at '{model_path}'.")
        st.error("Please ensure the 'rf_model.pkl' file is in the same directory as 'app.py'.")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        return None

def main():
    """
    Main function to define the Streamlit application's UI and logic.
    """

    # --- Custom CSS for Professional Look ---
    st.markdown(
        """
        <style>
        .main-title {
            font-size: 2.8rem;
            font-weight: 700;
            color: #2d3748;
            margin-bottom: 0.2em;
            text-align: center;
            letter-spacing: 1px;
        }
        .subtitle {
            font-size: 1.2rem;
            color: #4a5568;
            text-align: center;
            margin-bottom: 2em;
        }
        .stButton > button {
            background-color: #e3f6fd;
            color: #2d3748;
            font-weight: 600;
            border-radius: 8px;
            padding: 0.5em 2em;
            margin-top: 1em;
            border: 1px solid #b5c9d6;
            box-shadow: 0 2px 8px rgba(58,80,107,0.04);
        }
        .stButton > button:hover {
            background-color: #b5c9d6;
            color: #22223b;
        }
        .st-expanderHeader {
            font-size: 1.1rem;
            color: #2d3748;
        }
        .stApp {
            background: linear-gradient(135deg, #e3f6fd 0%, #f7faff 100%);
        }
        /* Improve input field contrast */
        input, .stNumberInput input, .stTextInput input, .stDateInput input, .stTimeInput input {
            background-color: #ffffff !important;
            color: #2d3748 !important;
            border: 1px solid #b5c9d6 !important;
        }
        label, .st-cb, .st-bb, .stNumberInput label, .stTextInput label, .stDateInput label, .stTimeInput label {
            color: #2d3748 !important;
            font-weight: 500;
        }
        .stNumberInput, .stTextInput, .stDateInput, .stTimeInput {
            margin-bottom: 1em !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="main-title">AQI Prediction</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">'
        'This application predicts the Air Quality Index (AQI) based on sensor readings, environmental factors, and time information.'
        '</div>', unsafe_allow_html=True
    )

    # Load the model
    model = load_model()

    # If the model isn't loaded, stop the app execution.
    if model is None:
        return

    # --- Feature Importance Section ---
    st.markdown("<h4 style='color:#3a506b; margin-bottom:0.5em;'>Model Insights</h4>", unsafe_allow_html=True)
    with st.expander("View Feature Importances"):
        try:
            # Define feature names in the same order as the training data
            feature_names = [
                'Tin oxide sensor (PT08.S1)', 'Titania sensor (PT08.S2)', 'Tungsten oxide sensor (PT08.S3)',
                'Tungsten oxide sensor (PT08.S4)', 'Indium oxide sensor (PT08.S5)', 'Temperature (°C)',
                'Relative Humidity (%)', 'Absolute Humidity', 'Year', 'Month', 'Day', 'Hour',
                'Day of Week', 'Week of Year'
            ]
            
            importances = model.feature_importances_
            
            feature_importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values(by='Importance', ascending=False)

            st.write("This chart shows how much each feature contributes to the model's predictions.")
            st.bar_chart(feature_importance_df.set_index('Feature'))

        except AttributeError:
            st.warning("The loaded model does not support feature importances.")
        except Exception as e:
            st.error(f"Could not display feature importances: {e}")


    st.markdown("<h4 style='color:#3a506b; margin-bottom:0.5em;'>Input Features</h4>", unsafe_allow_html=True)
    st.markdown("<div style='margin-bottom:1.5em'></div>", unsafe_allow_html=True)

    # Use columns for a cleaner layout on the main page
    col1, col2, col3 = st.columns([1.1, 1, 1])

    with col1:
        st.markdown("<div style='font-weight:600; color:#3a506b; margin-bottom:0.5em;'>Sensor Readings</div>", unsafe_allow_html=True)
        s1_co = st.number_input('Tin oxide sensor (PT08.S1)', help="Sensor reading targeted to CO", value=0.0, key='s1')
        s2_nmhc = st.number_input('Titania sensor (PT08.S2)', help="Sensor reading targeted to Non-Methane Hydrocarbons (NMHC)", value=0.0, key='s2')
        s3_nox = st.number_input('Tungsten oxide sensor (PT08.S3)', help="Sensor reading targeted to Nitrogen Oxides (NOx)", value=0.0, key='s3')
        s4_no2 = st.number_input('Tungsten oxide sensor (PT08.S4)', help="Sensor reading targeted to Nitrogen Dioxide (NO2)", value=0.0, key='s4')
        s5_o3 = st.number_input('Indium oxide sensor (PT08.S5)', help="Sensor reading targeted to Ozone (O3)", value=0.0, key='s5')

    with col2:
        st.markdown("<div style='font-weight:600; color:#3a506b; margin-bottom:0.5em;'>Environmental Factors</div>", unsafe_allow_html=True)
        temp = st.number_input("Temperature (°C)", value=25.0, key='temp')
        rh = st.number_input("Relative Humidity (%)", value=50.0, key='rh')
        ah = st.number_input("Absolute Humidity", value=1.0, key='ah')

    with col3:
        st.markdown("<div style='font-weight:600; color:#3a506b; margin-bottom:0.5em;'>Date and Time</div>", unsafe_allow_html=True)
        d = st.date_input("Date", datetime.date.today(), key='date')
        t = st.time_input("Time", datetime.time(12, 0), key='time')

    st.markdown("<div style='margin-bottom:2em'></div>", unsafe_allow_html=True)


    # Center the button using columns
    btn_col1, btn_col2, btn_col3 = st.columns([2, 1, 2])
    with btn_col2:
        predict_clicked = st.button("Predict AQI", use_container_width=True)

    if predict_clicked:
        # Combine date and time to extract features
        dt_object = datetime.datetime.combine(d, t)
        year = dt_object.year
        month = dt_object.month
        day = dt_object.day
        hour = dt_object.hour
        day_of_week = dt_object.weekday()  # Monday=0, Sunday=6
        week_of_year = dt_object.isocalendar().week

        # IMPORTANT: The order of features must match the order used during model training.
        user_features = [[
            s1_co, s2_nmhc, s3_nox, s4_no2, s5_o3,  # Sensor features
            temp, rh, ah,                            # Environmental features
            year, month, day, hour, day_of_week, week_of_year  # Time features
        ]]

        try:
            prediction = model.predict(user_features)

            st.markdown("<div style='margin-top:2em'></div>", unsafe_allow_html=True)
            st.markdown(
                f"<div style='background-color:#e3f6fd; border-radius:10px; padding:1.5em; text-align:center; margin-bottom:1em; border: 1px solid #b5c9d6;'>"
                f"<span style='font-size:1.5rem; color:#3a506b; font-weight:700;'>Predicted AQI: {prediction[0]:.2f}</span>"
                "</div>",
                unsafe_allow_html=True
            )
            st.info("Note: This prediction is based on the 14 input features provided above.")

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
    else:
        st.info("Enter the features above and click 'Predict AQI' to see the result.")

if __name__ == "__main__":
    main()
