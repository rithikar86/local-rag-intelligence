# Local RAG: Private PDF Assistant 🤖

A Retrieval-Augmented Generation (RAG) system that allows you to chat with any PDF locally. 
No data leaves your machine. No API keys required.

## 🚀 Key Features
- **100% Local:** Uses Ollama (Llama 3.2 & Nomic-Embed-Text).
- **Fast Search:** Uses FAISS for vector similarity search.
- **Modern UI:** Built with Streamlit for a clean user experience.

## 🛠️ Technical Stack
- **Framework:** LangChain
- **Vector DB:** FAISS (In-memory)
- **UI:** Streamlit
- **LLM/Embeddings:** Ollama

## 🔧 Installation
1. Install [Ollama](https://ollama.com/) and pull models:
   ```bash
   ollama pull llama3.2
   ollama pull nomic-embed-text