import streamlit as st
from openai import OpenAI
import pandas as pd

st.set_page_config(layout='wide')
st.markdown('<style>div.block-container{padding-top:2rem;}</style>',unsafe_allow_html=True)
st.title("🤩😡😶 Sentiment Analysis!")
col1,col2 = st.columns(2)
with col1:
 api_key = st.text_input("Enter your OpenAI API key", type="password")
 if api_key:
  client = OpenAI(api_key=api_key)

  #Creating Assistant
  assistant = client.beta.assistants.create(
   name= "Sentiment Analyzer",
   instructions="You are a Sentiment Analysis agent who gets a comment or review and claassify it based on the sentiments, Positive,Negative and Neutral, also answer user's questions about comments_df",
   model = "gpt-4o",
   tools=[{"type": "code_interpreter"}]                                         
        )
  st.markdown("<h1 style='font-size:20px;'> Load Comment Dataset:</h1>", unsafe_allow_html=True)
  uploaded_file = st.file_uploader("")

  if uploaded_file is not None:
    
      if st.button('Analyze the Sentiment!'):
        comments_df=pd.read_csv(uploaded_file)
        comments_df = comments_df.iloc[:1]

        def analyze_comment(comment):
            #Creating Thread
            thread = client.beta.threads.create()

            #Create message

            message = client.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content=comment                                   
            )

            #Create the Run  

            run = client.beta.threads.runs.create(
                thread_id=thread.id,
                assistant_id=assistant.id
            )

            while run.status != 'completed':
              run_status = client.beta.threads.runs.retrieve(
                  thread_id=thread.id,
                  run_id=run.id
                  )
              #print("Run Status:",keep_retrieving_run.status)
              if run_status.status == "completed":
                    #print("\n")
                    break

            all_messages = client.beta.threads.messages.list(
                thread_id=thread.id
            )
            assistant_response = all_messages.data[0].content[0].text.value
            response_text = assistant_response.lower().strip()
            if "positive" in response_text:
                sentiment = "Positive"
            elif "negative" in response_text:
                sentiment = "Negative"
            elif "neutral" in response_text:
                sentiment = "Neutral"
            else:
                sentiment = "Unclassified"
            
            return sentiment, assistant_response

        results = comments_df['Comment'].apply(analyze_comment)

        # Create separate columns for Sentiment and Explanation
        comments_df['Sentiment'] = results.apply(lambda x: x[0])
        comments_df['Explanation'] = results.apply(lambda x: x[1])
        st.session_state.comments_df = comments_df  # Store DataFrame in session state
        st.write(comments_df)

with col2:
  if api_key:
    st.markdown("<h1 style='font-size:20px;'> Chat with the Assistant:</h1>", unsafe_allow_html=True)
    # Access the DataFrame from session state in the chat
    if 'comments_df' in st.session_state:
            comments_df = st.session_state.comments_df
            st.write("Analyzed DataFrame is available for the assistant.")
    if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
    user_input = st.text_area("Enter your question or comment here:")
    if st.button('Ask the Assistant'):
        if user_input:
            # Creating Thread for user input
            thread = client.beta.threads.create()

            # Create message for user input
            context = ""
            if 'comments_df' in st.session_state:
                context = f"The analyzed comments dataframe is: {st.session_state.comments_df.to_string(index=False)}"

            # Create message for user input
            message = client.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content=user_input + "\n\n" + context
            )

            # Create the Run for user input
            run = client.beta.threads.runs.create(
                thread_id=thread.id,
                assistant_id=assistant.id
            )

            while run.status != 'completed':
                run_status = client.beta.threads.runs.retrieve(
                    thread_id=thread.id,
                    run_id=run.id
                )
                if run_status.status == "completed":
                    break

            all_messages = client.beta.threads.messages.list(
                thread_id=thread.id
            )
            assistant_response = all_messages.data[0].content[0].text.value
            st.session_state.chat_history.append(("User", user_input))
            st.session_state.chat_history.append(("Assistant", assistant_response))
            
    
    # Display chat history
    for role, msg in reversed(st.session_state.chat_history):
            st.write(f"**{role}:** {msg}")
            st.write("---")
    
