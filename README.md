# AQI Prediction Streamlit Application

This is an interactive web application built with Streamlit that predicts the Air Quality Index (AQI) based on sensor readings, environmental factors, and time-based features. The application uses a pre-trained Random Forest Regressor model to make predictions in real-time.

 
*(Note: You can replace the above URL with a link to a screenshot of your application.)*

---

## Features

-   **Real-time AQI Prediction**: Input 14 different features and get an instant AQI prediction.
-   **Interactive UI**: User-friendly interface with columns for organized input of sensor, environmental, and time data.
-   **Model Insights**: A collapsible section displays a bar chart of the model's feature importances, helping to understand which factors most influence the prediction.
-   **Efficient Model Loading**: The machine learning model is cached on first load to ensure the application remains fast and responsive.
-   **Robust Error Handling**: The application gracefully handles potential errors, such as a missing model file.

---

## How It Works

The application consists of two main components:

1.  **Frontend**: The user interface is built entirely in Python using the **Streamlit** library. It provides input widgets (number inputs, date/time pickers) for the user to enter feature values.
2.  **Backend/Model**: A pre-trained Random Forest model, saved as `rf_model.pkl`, is loaded using `joblib`. When the user clicks "Predict AQI", the input features are sent to the model, which returns the predicted AQI value.

---

## Setup and Installation

To run this application on your local machine, please follow these steps:

**1. Clone the Repository**
```bash
git clone https://github.com/your-username/AQI-Prediction.git
cd AQI-Prediction
```

**2. Create a Virtual Environment (Recommended)**
It's a good practice to create a virtual environment to manage project-specific dependencies.
```bash
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

**3. Install Dependencies**
Install all the required Python packages using the `requirements.txt` file.
```bash
pip install -r requirements.txt
```

**4. Place the Model File**
Ensure that your pre-trained model file, `rf_model.pkl`, is present in the root directory of the project.

**5. Run the Streamlit App**
Execute the following command in your terminal:
```bash
streamlit run app.py
```
The application should now open in a new tab in your web browser.

---

## Usage

1.  Open the application in your browser.
2.  Use the input widgets to enter values for the 14 features across the three columns:
    -   **Sensor Readings**
    -   **Environmental Factors**
    -   **Date and Time**
3.  Click the **"Predict AQI"** button to see the result.
4.  The predicted AQI will be displayed in a success box.
5.  Optionally, expand the **"View Feature Importances"** section to see a chart of which features are most influential to the model.

---

## File Structure

```
.
├── app.py              # The main Streamlit application script
├── rf_model.pkl        # The pre-trained machine learning model
├── requirements.txt    # A list of Python packages required to run the app
└── README.md           # This file
```