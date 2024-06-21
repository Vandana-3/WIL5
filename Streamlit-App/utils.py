import pandas as pd
import streamlit as st
import joblib

# Load the trained model
model = joblib.load('model/risk_model.pkl')

def preprocess_input(data, model):
    data_encoded = pd.get_dummies(data)
    return data_encoded.reindex(columns=model.feature_importances_.argsort(), fill_value=0)

def home_page():
    st.title('Compliance Intelligence and Risk Management System (CIRMS)')
    st.subheader('Empowering Efficient and Effective Regulatory Oversight')
    
    # Welcome Message
    st.write("""
    Welcome to CIRMS! Our tool leverages AI to help regulators prioritize inspections and enhance compliance efforts.
    """)

    # Quick Access Links
    st.write("## Quick Actions")
    st.markdown("[Start a Risk Assessment](#Inspection) | [Start an Eligibility Assessment](#Licensing)")

    # Quick Start Guide
    st.write("## Quick Start Guide")
    st.write("""
    1. Navigate to **Inspection** to assess entity risk levels.
    2. Use the **Data Dashboard** to view and analyze overall compliance data, risk levels, inspection outcomes, and other key metrics.
    3. Explore **Licensing** for checking the eligibility of entities requesting permits.
    4. Utilize **Compliance** to access functionalities for predicting potential future infractions and recommending effective, customized actions for each entity.
    5. Engage with the **Chatbot** to answer questions about regulatory laws or specific entities.
    """)

    # Announcements and Updates
    st.write("## Latest News")
    st.write("""
    - **New Feature**: Enhanced risk prediction model now available!
    - **Coming Soon**: Compliance recommendations module.
    """)

    # Contact and Support
    st.write("## Contact and Support")
    st.write("""
    - **Email**: support@example.com
    - **Documentation**: [User Guide](#)
    """)

def licensing_section():
    st.title('Licensing')
    st.write("This section is under development and will include functionalities for checking the eligibility of entities requesting permits.")

def compliance_section():
    st.title('Compliance')
    st.write("This section is under development and will include functionalities for predicting potential future infractions and recommending effective, customized actions for each entity.")

def chatbot_section():
    st.title('Intelligent Chatbot')
    st.write("This section is under development and will include an intelligent chatbot to answer questions about regulatory laws or specific entities.")
