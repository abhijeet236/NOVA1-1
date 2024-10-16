# NOVA1-1
NOVA is an intelligent chatbot system designed to assist employees of large public sector organizations by answering queries related to HR policies, IT support, and organizational events. NOVA uses a combination of local document processing and the Groq LLM API for generating accurate responses from multiple uploaded PDFs. The chatbot also integrates GPU support for fast document embedding and similarity search using FAISS and PyTorch.

Features
Multiple PDF Upload: Handles and processes multiple PDFs for querying.
Question-Answering: Responds to queries based on uploaded PDF documents.
GPU Acceleration: Utilizes the RTX 3050 GPU for embedding generation and similarity searches using FAISS.
Groq API Integration: Sends queries to the Groq LLM for generating human-like responses.
Admin-Only PDF Upload: Restricts PDF upload functionality to registered administrators.
ChatGPT-like Interface: Clean and modern user interface resembling ChatGPT, with a two-dialogue box design.
Error Handling: Provides a fallback message when a query is out of context.
Technologies Used
Python: Backend logic.
Streamlit: Web interface.
PyTorch: GPU-based model inference.
FAISS: Fast similarity search with GPU support.
PyPDF2: PDF processing and text extraction.
Groq API: Language model API for advanced question answering.
Requirements
Python 3.8+
PyTorch (with CUDA support for GPU acceleration)
FAISS (with GPU support)
CUDA 11.x (for GPU operations on RTX 3050)
Groq API Key (Required for querying the language model)
Streamlit (For the web interface)
