import streamlit as st
import openai
import speech_recognition as sr
from gtts import gTTS
from io import BytesIO
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Set your OpenAI API key
openai.api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")

# Sidebar information
st.sidebar.markdown("## Author: Bushra Saleem")
st.sidebar.markdown("### Assignment by: HTS")
st.sidebar.markdown("[GitHub](https://github.com/BushraSaleem23)")
st.sidebar.markdown("[LinkedIn](https://www.linkedin.com/in/bushra-saleem-4250a8290/)")

# Set the page title and header
st.title("Urdu Voice Chatbot using RAG")
st.header("Ask your question in Urdu and get a response in both text and audio!")

# Load PDF file
@st.cache_resource
def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    split_docs = text_splitter.split_documents(documents)
    return split_docs

# Load your PDF file here
pdf_file_path = ("Revised Manual for School Based Assessment-2024_1.pdf")  
documents = load_pdf(pdf_file_path)

# Create embeddings and vector store
embeddings = OpenAIEmbeddings()
vector_store = FAISS.from_documents(documents, embeddings)
retrieval_chain = RetrievalQA.from_chain_type(llm=openai.ChatCompletion.create, chain_type="stuff", retriever=vector_store.as_retriever())

# Function to recognize speech and convert it to text
def speech_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening... Please ask your question in Urdu.")
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio, language="ur")
            return text
        except sr.UnknownValueError:
            st.error("Sorry, I couldn't understand that.")
            return ""
        except sr.RequestError:
            st.error("Could not request results. Check your internet connection.")
            return ""

# Function to convert text to speech
def text_to_speech(text):
    tts = gTTS(text, lang='ur')
    audio_file = BytesIO()
    tts.write_to_fp(audio_file)
    audio_file.seek(0)
    return audio_file

# Function to generate response using RAG
def generate_response(question):
    result = retrieval_chain({"query": question})
    return result['result']

# Main chatbot functionality
def chatbot():
    # Display a button to record audio input
    st.write("### Record Your Question in Urdu")
    if st.button("🎙️ Click to Record Audio"):
        user_input = speech_to_text()
        if user_input:
            st.success(f"You said: {user_input}")
        else:
            st.error("No audio detected.")
    else:
        # Option to type the question manually
        user_input = st.text_input("Or type your question in Urdu:")

    if user_input:
        # Generate response using RAG
        with st.spinner("Generating response..."):
            response = generate_response(user_input)

        # Display the response in text
        st.write("Chatbot response (text):", response)

        # Convert response to audio and play it
        audio_file = text_to_speech(response)
        st.audio(audio_file, format="audio/mp3")

# Run the chatbot app
if __name__ == "__main__":
    chatbot()


