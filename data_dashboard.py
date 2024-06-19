import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

def data_dashboard_section():
    st.title('Data Dashboard')
    st.header('Overview')

    # Overall Compliance Rate
    st.subheader('Compliance Rate Over Time')
    compliance_data = pd.DataFrame({
        'Date': pd.date_range(start='2023-01-01', periods=12, freq='M'),
        'Compliance Rate': np.random.rand(12) * 100
    })
    fig_compliance = px.line(compliance_data, x='Date', y='Compliance Rate', title="Compliance Rate Over Time")
    st.plotly_chart(fig_compliance)

    # Risk Levels Distribution
    st.subheader('Risk Levels Distribution')
    risk_data = pd.DataFrame({
        'Risk Level': ['Low', 'Moderate', 'High'],
        'Count': [1500, 2500, 1000]
    })
    fig_risk = px.pie(risk_data, values='Count', names='Risk Level', title="Risk Levels Distribution")
    st.plotly_chart(fig_risk)

    # Inspection Outcomes
    st.subheader('Inspection Outcomes')
    inspection_data = pd.DataFrame({
        'Outcome': ['Pass', 'Fail'],
        'Count': [3000, 2000]
    })
    fig_inspection = px.bar(inspection_data, x='Outcome', y='Count', title="Inspection Outcomes")
    st.plotly_chart(fig_inspection)

    # Infraction Statistics
    st.subheader('Infraction Statistics Over Time')
    infraction_data = pd.DataFrame({
        'Date': pd.date_range(start='2023-01-01', periods=12, freq='M'),
        'Minor Infractions': np.random.randint(0, 50, 12),
        'Major Infractions': np.random.randint(0, 20, 12)
    })
    fig_infraction = px.line(infraction_data, x='Date', y=['Minor Infractions', 'Major Infractions'], title="Infraction Statistics Over Time")
    st.plotly_chart(fig_infraction)

    # Complaint Analysis
    st.subheader('Complaint Analysis Over Time')
    complaint_data = pd.DataFrame({
        'Date': pd.date_range(start='2023-01-01', periods=12, freq='M'),
        'Minor Complaints': np.random.randint(0, 100, 12),
        'Major Complaints': np.random.randint(0, 30, 12)
    })
    fig_complaint = px.line(complaint_data, x='Date', y=['Minor Complaints', 'Major Complaints'], title="Complaint Analysis Over Time")
    st.plotly_chart(fig_complaint)

    # Sentiment Analysis
    st.subheader('Public Sentiment Over Time')
    sentiment_data = pd.DataFrame({
        'Date': pd.date_range(start='2023-01-01', periods=12, freq='M'),
        'Sentiment Score': np.random.rand(12) * 100
    })
    fig_sentiment = px.line(sentiment_data, x='Date', y='Sentiment Score', title="Public Sentiment Over Time")
    st.plotly_chart(fig_sentiment)

    # License Applications and Approvals
    st.subheader('License Applications and Approvals')
    license_data = pd.DataFrame({
        'Month': pd.date_range(start='2023-01-01', periods=12, freq='M').strftime('%B'),
        'Applications': np.random.randint(50, 150, 12),
        'Approvals': np.random.randint(30, 100, 12),
        'Rejections': np.random.randint(10, 50, 12)
    })
    fig_license = px.bar(license_data, x='Month', y=['Applications', 'Approvals', 'Rejections'], title="License Applications and Approvals")
    st.plotly_chart(fig_license)

    # Enforcement Actions
    st.subheader('Enforcement Actions')
    enforcement_data = pd.DataFrame({
        'Action Type': ['Fines', 'Warnings', 'Suspensions'],
        'Count': [100, 200, 50]
    })
    fig_enforcement = px.bar(enforcement_data, x='Action Type', y='Count', title="Enforcement Actions")
    st.plotly_chart(fig_enforcement)

    # Entity Performance
    st.subheader('Entity Performance Comparison')
    entity_performance_data = pd.DataFrame({
        'Entity ID': np.arange(1, 101),
        'Compliance Score': np.random.rand(100) * 100,
        'Risk Score': np.random.randint(1, 19, 100)
    })
    fig_entity_performance = px.scatter(entity_performance_data, x='Compliance Score', y='Risk Score', title="Entity Performance Comparison")
    st.plotly_chart(fig_entity_performance)
