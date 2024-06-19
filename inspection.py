import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from utils import preprocess_input

# Load the trained model
model = joblib.load('model/risk_model.pkl')

def inspection_section():
    st.title('Risk Prediction Tool')

    # Option to choose between manual input or file upload
    input_method = st.radio("Choose input method:", ("Manual Entry", "Upload CSV"))

    if input_method == "Manual Entry":
        # User inputs
        st.header('Enter Entity Details')

        # Autocomplete Text Box for Entity ID
        entity_id = st.text_input('Entity ID', help='Enter the unique ID of the entity')

        # Dropdown Menu for Number of Clients Served Annually
        num_clients = st.selectbox(
            'Number of Clients Served Annually',
            ['<200', '201-500', '>500'],
            help='Select the range of clients served annually'
        )

        # Dropdown Menu for Past Infraction History Type
        past_infraction_type = st.selectbox(
            'Past Infraction History Type',
            ['None', 'Minor Infractions', 'Major Infractions'],
            help='Select the type of past infractions'
        )

        # Date Picker for Past Infraction History Timeline
        past_infraction_timeline = st.selectbox(
            'Past Infraction History Timeline',
            ['None', 'Within past year', '1-3 years ago'],
            help='Select the timeline of past infractions'
        )

        # Dropdown Menu for Public Complaints Last Quarter
        public_complaints = st.selectbox(
            'Public Complaints Last Quarter',
            ['None', 'Minor', 'Major'],
            help='Select the severity of public complaints received in the last quarter'
        )

        # Dropdown Menu for Quarterly Public Sentiment Analysis
        sentiment_analysis = st.selectbox(
            'Quarterly Public Sentiment Analysis',
            ['None', 'Flagged'],
            help='Select the result of the quarterly public sentiment analysis'
        )

        # Radio Buttons for Previous Inspection Results
        inspection_results = st.radio(
            'Previous Inspection Results',
            ['Pass', 'Fail', 'None'],
            help='Select the result of the previous inspection'
        )

        # Create a dataframe for input
        input_data = pd.DataFrame({
            'Number of Clients Served Annually': [num_clients],
            'Past Infraction History Type': [past_infraction_type],
            'Past Infraction History Timeline': [past_infraction_timeline],
            'Public Complaints Last Quarter': [public_complaints],
            'Quarterly Public Sentiment Analysis': [sentiment_analysis],
            'Previous Inspection Results': [inspection_results]
        })

        # Preprocess the input data
        input_data_encoded = preprocess_input(input_data, model)

        # Predict risk score
        if st.button('Predict Risk Score'):
            risk_score = model.predict(input_data_encoded)[0]
            
            st.header('Data Summary')
            st.write(input_data.describe(include='all'))

            st.header('Prediction Results')
            st.write(f'Predicted Risk Score: {risk_score}')

    elif input_method == "Upload CSV":
        st.header('Upload Entity Data File')
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            entity_data = pd.read_csv(uploaded_file)
            st.write("Uploaded Data Preview:")
            st.dataframe(entity_data.head())

            # Preprocess the input data
            input_data_encoded = preprocess_input(entity_data, model)

            # Predict risk scores
            if st.button('Predict Risk Scores'):
                risk_scores = model.predict(input_data_encoded)
                entity_data['Predicted Risk Score'] = risk_scores
                
                st.header('Data Summary')
                st.write(entity_data.describe(include='all'))

                st.header('Prediction Results')

                # Ranking and Filtering Options
                rank_option = st.selectbox("Rank entities by:", ["Risk Score"])
                top_x_option = st.selectbox("Select top entities by:", ["Percentage", "Number"])
                if top_x_option == "Percentage":
                    top_x_value = st.slider("Select top X percentage:", min_value=1, max_value=100, value=10)
                    top_entities = entity_data.nlargest(int(len(entity_data) * top_x_value / 100), 'Predicted Risk Score')
                else:
                    top_x_value = st.slider("Select top X number:", min_value=1, max_value=len(entity_data), value=10)
                    top_entities = entity_data.nlargest(top_x_value, 'Predicted Risk Score')

                st.write(f"Top {top_x_value} entities based on {rank_option}:")
                st.dataframe(top_entities)

                # Download Options
                st.download_button(
                    label="Download data as CSV",
                    data=entity_data.to_csv(index=False),
                    file_name='risk_predictions.csv',
                    mime='text/csv'
                )
                st.download_button(
                    label="Download data as Excel",
                    data=entity_data.to_excel(index=False),
                    file_name='risk_predictions.xlsx',
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                )

    # Tabs for Data Summary and Visualization
    st.header('Analysis')
    tab1, tab2, tab3 = st.tabs(["Data Summary", "Risk Predictions", "Historical Trends"])

    with tab1:
        st.subheader('Data Summary')
        st.write(input_data.describe(include='all') if input_method == "Manual Entry" else entity_data.describe(include='all'))

    with tab2:
        st.subheader('Risk Predictions')
        if input_method == "Manual Entry" and st.button('Predict Risk Score'):
            st.write(f'Predicted Risk Score: {risk_score}')
        elif input_method == "Upload CSV" and st.button('Predict Risk Scores'):
            st.write(entity_data[['Entity ID', 'Predicted Risk Score']])
            st.dataframe(top_entities)

    with tab3:
        st.subheader('Historical Trends')
        # Example Data Visualization (using dummy data)
        dummy_data = pd.DataFrame({
            'Entity ID': np.arange(1, 101),
            'Risk Score': np.random.randint(1, 19, 100),
            'Risk Level': np.random.choice(['Low', 'Moderate', 'High'], 100)
        })

        # Risk Level Distribution Chart
        fig = px.histogram(dummy_data, x='Risk Level', title="Risk Levels Distribution")
        st.plotly_chart(fig)

        # Entity List and Details
        st.write("### Entity List")
        st.dataframe(dummy_data)

        selected_entity_id = st.selectbox("Select Entity ID to view details", dummy_data['Entity ID'])
        entity_details = dummy_data[dummy_data['Entity ID'] == selected_entity_id]
        st.write("### Entity Details")
        st.write(entity_details)


