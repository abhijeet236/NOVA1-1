import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from groq import Groq
import bcrypt
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Streamlit setup
st.set_page_config(page_title="Next Gen Organizational Virtual Assistant", layout="wide")
st.title("🤖 Next Gen Organizational Virtual Assistant")

# User Authentication Setup
st.sidebar.header("Login")
login_username = st.sidebar.text_input("Username")  # Corrected here
login_password = st.sidebar.text_input("Password", type="password")

# Dummy user data for demonstration purposes
# In a real application, fetch this from a secure database
users_db = {
    "admin": bcrypt.hashpw("adminpassword".encode('utf-8'), bcrypt.gensalt()),
}

# Function to authenticate users
def authenticate_user(username, password):
    hashed_pw = users_db.get(username)
    if hashed_pw and bcrypt.checkpw(password.encode('utf-8'), hashed_pw):
        return 'admin' if username == "admin" else 'user'
    return None

# Initialize session state variables
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'role' not in st.session_state:
    st.session_state.role = None
if 'faiss_index' not in st.session_state:
    st.session_state.faiss_index = None
if 'pdfs_processed' not in st.session_state:
    st.session_state.pdfs_processed = False

# Handle login
if st.sidebar.button("Login"):
    role = authenticate_user(login_username, login_password)
    if role == 'admin':
        st.session_state.logged_in = True
        st.session_state.role = 'admin'
        st.sidebar.success("Logged in as Admin.")
    elif role:
        st.session_state.logged_in = True
        st.session_state.role = 'user'
        st.sidebar.warning("Logged in as Regular User.")
    else:
        st.sidebar.error("Invalid credentials.")

# Sidebar for file upload (admin only)
if st.session_state.logged_in and st.session_state.role == 'admin':
    st.sidebar.header("Upload Organizational Documents")
    uploaded_files = st.sidebar.file_uploader("Choose PDF files (HR policies, IT support guides, etc.)", type="pdf", accept_multiple_files=True)

    # Function to read PDF content
    def read_pdf(file):
        reader = PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text() or ''
        return text

    # Function to process PDFs
    def process_pdfs(files):
        progress_bar = st.sidebar.progress(0)
        st.sidebar.write("Processing the PDFs...")

        try:
            # Fetch and process content
            documents = []
            for i, file in enumerate(files):
                content = read_pdf(file)
                if content:
                    documents.append(content)
                progress_bar.progress((i + 1) / len(files) * 0.5)  # 50% progress for fetching

            if not documents:
                raise ValueError("No valid content could be fetched from the provided PDFs.")

            # Split documents
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
            split_docs = splitter.create_documents(documents)
            progress_bar.progress(0.7)  # 70% progress after splitting

            # Generate embeddings
            embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            st.session_state.faiss_index = FAISS.from_documents(split_docs, embedding_model)

            progress_bar.progress(1.0)
            st.sidebar.success("Processing completed!")
            st.session_state.pdfs_processed = True

            # Display metadata in sidebar
            for file in files:
                st.sidebar.write(f"Uploaded File: {file.name}")
                st.sidebar.write(f"File Size: {file.size} bytes")
        except Exception as e:
            st.sidebar.error(f"An error occurred during analysis: {str(e)}")
            st.session_state.pdfs_processed = False
        finally:
            progress_bar.empty()

    # Button to start analysis
    if st.sidebar.button("Analyze"):
        if not uploaded_files:
            st.sidebar.error("Please upload at least one valid PDF file.")
        else:
            process_pdfs(uploaded_files)

# Question-Answer Interface
if st.session_state.logged_in:
    st.header("Ask Your Questions")
    query = st.text_input("Enter your HR, IT, or Onboarding question:")

    if st.button("Get Answer"):
        if not st.session_state.pdfs_processed:
            st.error("Please analyze the PDFs first.")
        elif not query:
            st.error("Please enter a question.")
        else:
            try:
                with st.spinner("Generating the answer..."):
                    # Retrieve relevant documents
                    results = st.session_state.faiss_index.similarity_search(query, k=4)

                    if results:
                        context = "\n".join([doc.page_content for doc in results])

                        # Use Groq API to generate the answer
                        groq_api_key = "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" # Replace with your Groq API key
                        client = Groq(api_key=groq_api_key)

                        completion = client.chat.completions.create(
                            model="llama3-8b-8192",
                            messages=[{
                                "role": "system",
                                "content": "You are an organizational assistant that answers HR, IT support, and onboarding questions based on the provided context from organizational documents."
                            }, {
                                "role": "user",
                                "content": f"Context: {context}\n\nPlease answer the question based on the given context. If the answer is not in the context, say 'I don't have enough information to answer this question accurately.'"
                            }],
                            temperature=0.5,
                            max_tokens=1024,
                            top_p=1,
                            stream=False,
                            stop=None
                        )

                        answer = completion.choices[0].message.content
                        st.markdown(f"**Question:** {query}")
                        st.markdown(f"**Answer:** {answer}")

                    else:
                        st.error("No relevant information found to answer the question.")
            except Exception as e:
                st.error(f"An error occurred while generating the answer: {str(e)}")

    # Display processing status
    if st.session_state.pdfs_processed:
        st.sidebar.success("PDFs have been processed. You can ask multiple questions without re-analyzing.")
    else:
        st.sidebar.info("Please upload PDFs and click 'Analyze' to process the documents.")
else:
    st.sidebar.info("Please log in as an admin to upload documents and ask questions.")
