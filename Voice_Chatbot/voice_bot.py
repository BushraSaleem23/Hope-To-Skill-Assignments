# Necessary Imports
import streamlit as st
import openai
import pyttsx3
import speech_recognition as sr
import os
from itertools import zip_longest
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

# Load environment variables from .env file if available
load_dotenv()

st.title("Urdu Voice Bot")

st.write("Welcome to Urdu Bot.How may I assist you?")

# Sidebar for OpenAI API key input
st.sidebar.title("API Configuration")
api_key = st.sidebar.text_input("Enter your OpenAI API Key:", type="password")
if api_key:
    openai.api_key = api_key
else:
    st.sidebar.warning("Please enter your OpenAI API key to proceed.")

# Initialize session state variables
if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

executor = ThreadPoolExecutor(max_workers=2)

# Function to use OpenAI API for response generation in Urdu
def generate_response(user_input, for_voice=False):
    try:
        st.write("Using OpenAI API for response.")
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=f"Respond in Urdu: {user_input}",
            max_tokens=50 if for_voice else 100,
            temperature=0.7
        )
        st.write(f"Response from API: {response}")
        return response.choices[0].text.strip() if response.choices else ''
    except openai.error.AuthenticationError:
        st.error("Invalid API Key. Please check your API key and try again.")
        return "Invalid API Key"
    except openai.error.RateLimitError:
        st.error("API rate limit exceeded. Please try again later.")
        return "Rate limit exceeded"
    except openai.error.OpenAIError as e:
        st.error(f"OpenAI API Error: {e}")
        return "OpenAI API Error"
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return "Unexpected error"

def build_message_list():
    zipped_messages = []
    for human_msg, ai_msg in zip_longest(st.session_state['past'], st.session_state['generated']):
        if human_msg is not None:
            zipped_messages.append(human_msg)
        if ai_msg is not None:
            zipped_messages.append(ai_msg)
    return zipped_messages

def capture_audio():
    st.info("🎤 Listening... Please speak your question now!")
    r = sr.Recognizer()
    with sr.Microphone() as source:
        audio = r.listen(source, timeout=10)
    st.empty()  # Clears the 'Listening...' message

    try:
        recognized_text = r.recognize_google(audio, language="ur")
        if recognized_text:
            st.success('Your audio is captured.')  # Display success message
        st.write(f"Recognized text: {recognized_text}")  # Debugging: show recognized text
        return recognized_text
    except sr.UnknownValueError:
        st.warning("Could not understand audio, please try again.")
        return ""
    except sr.RequestError:
        st.error("API unavailable. Please check your internet connection and try again.")
        return ""
# Defining the function  of text to speech in the bot using pyttsx3
def text_to_speech(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def display_message(message, is_user=False):
    if is_user:
        st.write(f"**You:** {message}")
    else:
        st.write(f"**Bot:** {message}")

# Display the conversation chain
if st.session_state['generated']:
    for ai_msg, user_msg in zip(st.session_state['generated'], st.session_state['past']):
        if user_msg:
            display_message(user_msg, is_user=True)
        if ai_msg:
            display_message(ai_msg)

# Create a layout for buttons in a single row
col1, col2 = st.columns([1, 1])

with col1:
    if st.button("Ask Question? 🔊"):
        with st.spinner('Listening and processing your query...'):
            user_input = capture_audio()
            if user_input:
                st.session_state.past.append(user_input)
                # Generate the response for voice
                output = generate_response(user_input, for_voice=True)

                # Ensure the text is added to the state and displayed before speaking
                st.session_state.generated.append(output)
                display_message(output)  # Display the bot response
                display_message(user_input, is_user=True)  # Display the user input

                # Call the text-to-speech function
                executor.submit(text_to_speech, output)
            else:
                st.error("No input detected. Please try again.")

        # Reset the button state
        st.experimental_rerun()

with col2:
    if st.button("Clear 🗑️"):
        st.session_state.past = []
        st.session_state.generated = []
        st.experimental_rerun()

st.sidebar.write("Author: Bushra Saleem")
st.sidebar.write('Assignment-7 HTS')

