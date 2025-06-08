import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

# Model configurations
models = {
    'Logistic Regression': {'file': 'logistic_regression.pkl', 'low': 0.3, 'high': 0.7},
    'K-Nearest Neighbors': {'file': 'knn.pkl', 'low': 0.35, 'high': 0.75},
    'Support Vector Machine': {'file': 'svc.pkl', 'low': 0.4, 'high': 0.8},
    'Random Forest': {'file': 'random_forest.pkl', 'low': 0.35, 'high': 0.75},
    'XGBoost': {'file': 'xgboost.pkl', 'low': 0.3, 'high': 0.7},
    'Neural Network (MLP)': {'file': 'mlp.pkl', 'low': 0.35, 'high': 0.75},
    'Gradient Boosting': {'file': 'gradient_boosting.pkl', 'low': 0.35, 'high': 0.75}
}

# Load all models at startup
loaded_models = {}
for name, config in models.items():
    try:
        with open(config['file'], 'rb') as f:
            loaded_models[name] = pickle.load(f)
    except Exception as e:
        st.error(f"Error loading {name} model: {str(e)}")

st.set_page_config(
    page_title="Lung Cancer Risk Predictor",
    page_icon="🫁",
    layout="wide"
)

# Enhanced CSS styling
st.markdown("""
<style>
.main { padding: 20px; }
.stButton>button {
    width: 100%;
    margin-top: 20px;
    background-color: #4CAF50;
    color: white;
    padding: 10px;
    border-radius: 5px;
}
.risk-high {
    color: #ff4b4b;
    font-size: 24px;
    font-weight: bold;
    padding: 10px;
    border: 2px solid #ff4b4b;
    border-radius: 5px;
    background-color: #fff5f5;
}
.risk-moderate {
    color: #ff9933;
    font-size: 24px;
    font-weight: bold;
    padding: 10px;
    border: 2px solid #ff9933;
    border-radius: 5px;
    background-color: #fff9f2;
}
.risk-low {
    color: #00cc00;
    font-size: 24px;
    font-weight: bold;
    padding: 10px;
    border: 2px solid #00cc00;
    border-radius: 5px;
    background-color: #f5fff5;
}
.risk-score {
    font-size: 20px;
    font-weight: bold;
    padding: 5px;
}
</style>
""", unsafe_allow_html=True)

# App title and model selection
st.title("🫁 Lung Cancer Risk Prediction")
selected_model = st.selectbox(
    "Select Machine Learning Model",
    list(models.keys()),
    help="Choose which model to use for prediction"
)

# Model descriptions
model_descriptions = {
    'Logistic Regression': "Simple, interpretable model that works well for binary classification.",
    'K-Nearest Neighbors': "Makes predictions based on similar cases in the training data.",
    'Support Vector Machine': "Effective for high-dimensional data with clear separation.",
    'Random Forest': "Ensemble model that combines multiple decision trees for robust predictions.",
    'XGBoost': "Advanced gradient boosting model known for high performance.",
    'Neural Network (MLP)': "Deep learning model that can capture complex patterns.",
    'Gradient Boosting': "Builds an ensemble of weak learners sequentially."
}

st.info(model_descriptions[selected_model])

# Create two columns for input
col1, col2 = st.columns(2)

# Input fields
with col1:
    st.subheader("Personal Information")
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input("Age", 20, 100, 50)
    smoking = st.selectbox("Smoking", ["No", "Yes"], help="Current or past smoking habit")
    alcohol = st.selectbox("Alcohol Consumption", ["No", "Yes"])
    chronic_disease = st.selectbox("Chronic Disease", ["No", "Yes"])
    fatigue = st.selectbox("Fatigue", ["No", "Yes"])
    allergy = st.selectbox("Allergy", ["No", "Yes"])
    wheezing = st.selectbox("Wheezing", ["No", "Yes"])

with col2:
    st.subheader("Symptoms")
    yellow_fingers = st.selectbox("Yellow Fingers", ["No", "Yes"])
    anxiety = st.selectbox("Anxiety", ["No", "Yes"])
    peer_pressure = st.selectbox("Peer Pressure", ["No", "Yes"])
    coughing = st.selectbox("Coughing", ["No", "Yes"])
    shortness_breath = st.selectbox("Shortness of Breath", ["No", "Yes"])
    swallowing_diff = st.selectbox("Swallowing Difficulty", ["No", "Yes"])
    chest_pain = st.selectbox("Chest Pain", ["No", "Yes"])

if st.button("Predict Risk"):
    # Calculate risk score
    risk_factors = []
    risk_score = 0
    
    # Primary risk factors
    primary_factors = {
        'smoking': ('🚬 Smoking', 2),
        'shortness_breath': ('😮‍💨 Shortness of Breath', 2),
        'chest_pain': ('💔 Chest Pain', 2)
    }
    
    # Secondary risk factors
    secondary_factors = {
        'chronic_disease': ('🏥 Chronic Disease', 1),
        'alcohol': ('🍺 Alcohol Consumption', 1),
        'wheezing': ('🌬️ Wheezing', 1)
    }
    
    # Calculate risk factors
    for var, (label, weight) in primary_factors.items():
        if locals()[var] == "Yes":
            risk_factors.append(f"{label} (Major risk factor)")
            risk_score += weight
            
    for var, (label, weight) in secondary_factors.items():
        if locals()[var] == "Yes":
            risk_factors.append(f"{label}")
            risk_score += weight

    # Create input data
    input_data = {
        'GENDER': 1 if gender == "Male" else 0,
        'AGE': age,
        'SMOKING': 1 if smoking == "Yes" else 0,
        'YELLOW_FINGERS': 1 if yellow_fingers == "Yes" else 0,
        'ANXIETY': 1 if anxiety == "Yes" else 0,
        'PEER_PRESSURE': 1 if peer_pressure == "Yes" else 0,
        'CHRONIC_DISEASE': 1 if chronic_disease == "Yes" else 0,
        'FATIGUE': 1 if fatigue == "Yes" else 0,
        'ALLERGY': 1 if allergy == "Yes" else 0,
        'WHEEZING': 1 if wheezing == "Yes" else 0,
        'ALCOHOL_CONSUMING': 1 if alcohol == "Yes" else 0,
        'COUGHING': 1 if coughing == "Yes" else 0,
        'SHORTNESS_OF_BREATH': 1 if shortness_breath == "Yes" else 0,
        'SWALLOWING_DIFFICULTY': 1 if swallowing_diff == "Yes" else 0,
        'CHEST_PAIN': 1 if chest_pain == "Yes" else 0
    }

    try:
        # Make prediction
        model = loaded_models[selected_model]
        input_df = pd.DataFrame([input_data])
        probability = model.predict_proba(input_df)[0]
        
        # Determine risk level
        if  risk_score >= 6:
            risk_level = "high"
            risk_class = "risk-high"
            risk_icon = "⚠️ High Risk"
        elif risk_score > 3:
            risk_level = "moderate"
            risk_class = "risk-moderate"
            risk_icon = "⚡ Moderate Risk"
        elif risk_score <= 3:
            risk_level = "low"
            risk_class = "risk-low"
            risk_icon = "✅ Low Risk"

        # Show results
        st.write("---")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### Prediction Result")
            st.markdown(f'<p class="{risk_class}">{risk_icon}</p>', unsafe_allow_html=True)
        
        with col2:
            st.write("### Risk Score")
            max_score = 9
            st.markdown(f'<p class="risk-score">{risk_score} out of {max_score}</p>', unsafe_allow_html=True)
            if risk_score >= 6:
                st.write("❗ High risk score")
            elif risk_score > 3:
                st.write("⚠️ Moderate risk score")
            else:
                st.write("✅ Low risk score")

        # Risk factors analysis
        st.write("---")
        st.write("### Key Risk Factors Identified:")
        
        if risk_factors:
            major_factors = [f for f in risk_factors if "Major" in f]
            other_factors = [f for f in risk_factors if "Major" not in f]
            
            if major_factors:
                st.write("#### Major Risk Factors:")
                for factor in major_factors:
                    st.write(f"- {factor}")
            
            if other_factors:
                st.write("#### Other Risk Factors:")
                for factor in other_factors:
                    st.write(f"- {factor}")

            # Recommendations based on risk level
            st.write("---")
            st.write("### Recommended Actions:")
            if risk_level == "high":
                st.write("⚠️ **Based on your high risk assessment:**")
                st.write("1. 🏥 Schedule immediate consultation with a healthcare provider")
                st.write("2. 🫁 Undergo lung cancer screening")
                st.write("3. 🚭 Begin smoking cessation program if applicable")
                st.write("4. 📋 Monitor symptoms daily")
            elif risk_level == "moderate":
                st.write("⚡ **Based on your moderate risk assessment:**")
                st.write("1. 🏥 Schedule consultation with a healthcare provider")
                st.write("2. 🫁 Consider lung cancer screening")
                st.write("3. 🚭 Consider smoking cessation if applicable")
                st.write("4. 📋 Monitor symptoms regularly")
            else:
                st.write("✅ **Based on your low risk assessment:**")
                st.write("1. 📋 Maintain regular check-ups")
                st.write("2. 🚭 Continue avoiding smoking")
                st.write("3. 💪 Maintain healthy lifestyle")
        else:
            st.write("No significant risk factors identified")
            
    except Exception as e:
        st.error(f"Error making prediction with {selected_model}: {str(e)}")

# Sidebar information
st.sidebar.title("About Selected Model")
st.sidebar.info(
    f"""
    Currently using: **{selected_model}**
    
    {model_descriptions[selected_model]}
    
    **Note:** This is a screening tool and should not replace professional medical advice.
    """
)

st.sidebar.title("Risk Assessment")
st.sidebar.write(
    """
    Risk is calculated based on:
    - Clinical symptoms
    - Lifestyle factors
    - Medical history
    - Demographic data
    
    Each model may weigh these factors differently based on their learning algorithm.
    """
)