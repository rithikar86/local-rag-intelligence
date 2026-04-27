import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.chains import RetrievalQA
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="PDF Intelligence", page_icon="🧠", layout="wide")


# --- CUSTOM CSS FOR DESIGN ---
def local_css():
    st.markdown("""
    <style>
        /* Main Background */
        .stApp {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        }

        /* Chat Bubble Styling */
        .stChatMessage {
            border-radius: 15px;
            padding: 15px;
            margin-bottom: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        }

        /* Sidebar Styling */
        section[data-testid="stSidebar"] {
            background-color: #1e1e2f !important;
            color: white;
        }

        /* Custom Header */
        .main-header {
            font-family: 'Inter', sans-serif;
            color: #2c3e50;
            text-align: center;
            font-weight: 800;
            font-size: 3rem;
            margin-bottom: 0px;
        }

        .sub-header {
            text-align: center;
            color: #5d6d7e;
            margin-bottom: 2rem;
        }
    </style>
    """, unsafe_allow_html=True)


# Apply CSS
local_css()

# --- HEADER ---
st.markdown('<p class="main-header">PDF Intelligence Portal</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Local RAG System • Secure • Private</p>', unsafe_allow_html=True)

# --- SIDEBAR SETUP ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=80)
    st.title("Control Center")
    uploaded_file = st.file_uploader("📂 Upload PDF Knowledge Base", type="pdf")
    st.divider()
    st.info("System Status: **Ollama Active**")
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# --- RAG LOGIC ---
if uploaded_file:
    # Ensure data directory exists
    if not os.path.exists("data"):
        os.makedirs("data")

    file_path = os.path.join("data", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())


    # Cache the processing so it doesn't re-run every time you type a message
    @st.cache_resource
    def initialize_rag(file_path):
        # Step 1: Load and Split
        loader = PyPDFLoader(file_path)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = text_splitter.split_documents(loader.load())

        # Step 2: Embed using Nomic
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        vector_db = FAISS.from_documents(chunks, embeddings)

        # Step 3: Setup LLM (Llama 3.2)
        llm = ChatOllama(model="llama3.2", temperature=0)

        return RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_db.as_retriever()
        )


    # Initialize the chain
    try:
        qa_chain = initialize_rag(file_path)
        st.success(f"Successfully processed: {uploaded_file.name}")

        # --- CHAT INTERFACE ---
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display Chat History
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat Input
        if prompt := st.chat_input("Ask something about the document"):
            # User Message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Assistant Message
            with st.chat_message("assistant"):
                with st.spinner("Analyzing Knowledge Base..."):
                    try:
                        response = qa_chain.invoke(prompt)
                        answer = response["result"]
                        st.markdown(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                    except Exception as e:
                        st.error(f"Error generating response: {e}")

    except Exception as e:
        st.error(f"Initialization Error: {e}")
        st.info("Make sure you have run 'ollama pull llama3.2' and 'ollama pull nomic-embed-text'")

else:
    st.warning("👈 Please upload a PDF in the sidebar to begin.")