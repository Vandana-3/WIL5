import streamlit as st
from inspection import inspection_section
from data_dashboard import data_dashboard_section
from utils import home_page, licensing_section, compliance_section, chatbot_section

def main():
    st.sidebar.title("Menu")
    page = st.sidebar.radio(
        "",
        ["Home", "Licensing", "Inspection", "Compliance", "Data Dashboard", "Chatbot"],
        format_func=lambda page: {
            "Home": "🏠 Home",
            "Licensing": "📜 Licensing",
            "Inspection": "🔍 Inspection",
            "Compliance": "✅ Compliance",
            "Data Dashboard": "📊 Data Dashboard",
            "Chatbot": "🤖 Chatbot"
        }[page]
    )

    if page == "Home":
        home_page()
    elif page == "Licensing":
        licensing_section()
    elif page == "Inspection":
        inspection_section()
    elif page == "Compliance":
        compliance_section()
    elif page == "Data Dashboard":
        data_dashboard_section()
    elif page == "Chatbot":
        chatbot_section()

if __name__ == "__main__":
    main()
