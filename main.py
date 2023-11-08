import streamlit as st
import youtube_reader as ytb
import textwrap

st.title("YouTube Insight Guru")
st.image("https://cdn-icons-png.flaticon.com/512/1384/1384060.png", width=50)

with st.sidebar:
    with st.form(key='user_input_form'):
        youtube_url_input = st.sidebar.text_area(
            label="Enter the YouTube video URL:",
            max_chars=100
        )
        query_input = st.sidebar.text_area(
            label="Enter your question about the video:",
            max_chars=50,
            key="query_input"
        )
        openai_api_key_input = st.sidebar.text_input(
            label="Enter your OpenAI API Key:",
            key="openai_api_key",
            max_chars=100,
            type="password"
        )

        if st.form_submit_button(label='Submit'):
            pass 

if query_input and youtube_url_input:
    if not openai_api_key_input:
        st.info("Please provide your OpenAI API key to continue.")
        st.stop()
    else:
        video_db = ytb.create_custom_db_from_youtube_video_url(youtube_url_input)
        
        response_text, related_docs = ytb.get_custom_response_from_query(video_db, query_input)
        
     
        st.subheader("Answer:")
        st.text(textwrap.fill(response_text, width=85))
