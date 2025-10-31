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
    st.title('AQI Prediction System')
    st.markdown("""
    This application predicts the Air Quality Index (AQI) based on sensor readings,
    environmental factors, and time information.
    """)

    # Load the model
    model = load_model()

    # If the model isn't loaded, stop the app execution.
    if model is None:
        return

    # --- Feature Importance Section ---
    st.header("Model Insights")
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


    st.header("Input Features")
    st.markdown("<div style='margin-bottom:1.5em'></div>", unsafe_allow_html=True)

    # Use columns for a cleaner layout on the main page
    col1, col2, col3 = st.columns([1.1, 1, 1])

    with col1:
        st.subheader("Sensor Readings")
        s1_co = st.number_input('Tin oxide sensor (PT08.S1)', help="Sensor reading targeted to CO", value=0.0, key='s1')
        s2_nmhc = st.number_input('Titania sensor (PT08.S2)', help="Sensor reading targeted to Non-Methane Hydrocarbons (NMHC)", value=0.0, key='s2')
        s3_nox = st.number_input('Tungsten oxide sensor (PT08.S3)', help="Sensor reading targeted to Nitrogen Oxides (NOx)", value=0.0, key='s3')
        s4_no2 = st.number_input('Tungsten oxide sensor (PT08.S4)', help="Sensor reading targeted to Nitrogen Dioxide (NO2)", value=0.0, key='s4')
        s5_o3 = st.number_input('Indium oxide sensor (PT08.S5)', help="Sensor reading targeted to Ozone (O3)", value=0.0, key='s5')

    with col2:
        st.subheader("Environmental Factors")
        temp = st.number_input("Temperature (°C)", value=25.0, key='temp')
        rh = st.number_input("Relative Humidity (%)", value=50.0, key='rh')
        ah = st.number_input("Absolute Humidity", value=1.0, key='ah')

    with col3:
        st.subheader("Date and Time")
        d = st.date_input("Date", datetime.date.today(), key='date')
        t = st.time_input("Time", datetime.time(12, 0), key='time')

    st.write("")  # Add a little vertical space


    # A button to trigger the prediction
    if st.button("Predict AQI"):
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
            st.markdown("""---""")
            prediction = model.predict(user_features)

            st.subheader("Prediction Result")
            st.success(f"Predicted AQI: **{prediction[0]:.2f}**")
            st.info("Note: This prediction is based on the 14 input features provided above.")

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
    else:
        st.info("Enter the features above and click 'Predict AQI' to see the result.")

if __name__ == "__main__":
    main()
