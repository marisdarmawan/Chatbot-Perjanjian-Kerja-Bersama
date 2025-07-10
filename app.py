import os
import streamlit as st
# DIUBAH: Impor model dari Google (untuk embedding) dan OpenAI (untuk chat)
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# --- Konfigurasi Awal ---
st.set_page_config(page_title="Chatbot PKB PLN", page_icon="⚡", layout="wide")

# DIUBAH: Muat OPENROUTER_API_KEY dari secrets Streamlit
# Kita akan menyimpannya ke environment variable yang dikenali oleh library OpenAI
if 'OPENROUTER_API_KEY' not in os.environ:
    try:
        # Menggunakan nama secret yang baru
        openrouter_key = st.secrets["OPENROUTER_API_KEY"]
        # Library LangChain/OpenAI secara default mencari 'OPENAI_API_KEY'
        os.environ["OPENAI_API_KEY"] = openrouter_key
    except Exception as e:
        st.error("Harap atur OPENROUTER_API_KEY Anda di Streamlit secrets. Error: " + str(e))
        st.stop()
else:
    # Jika sudah ada, pastikan tetap diset ke env var yang benar
    os.environ["OPENAI_API_KEY"] = os.environ["OPENROUTER_API_KEY"]


# --- Fungsi-Fungsi Inti dengan Cache ---

@st.cache_resource
def load_models_and_vector_store():
    """
    Memuat model LLM, embeddings, dan vector store dari file lokal.
    """
    try:
        # PENTING: Model embedding tetap menggunakan Google karena database dibuat dengan ini
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

        # Muat vector store dari file lokal
        vector_store = FAISS.load_local(
            "pkb_pln_faiss_index",
            embeddings,
            allow_dangerous_deserialization=True
        )

        # DIUBAH: Inisialisasi model Chat menggunakan OpenRouter
        llm = ChatOpenAI(
            model_name="meta-llama/llama-4-maverick:free", # Model gratis dari OpenRouter
            temperature=0.2,
            openai_api_base="https://openrouter.ai/api/v1", # Mengarahkan ke server OpenRouter
            default_headers={
                # GANTI DENGAN URL APLIKASI STREAMLIT ANDA
                "HTTP-Referer": "https://chatbot-perjanjian-kerja-bersama.streamlit.app",
                "X-Title": "Chatbot PKB PLN",
            }
        )
        
        return llm, vector_store
    except Exception as e:
        st.error(f"Gagal memuat model atau vector store. Pastikan folder 'pkb_pln_faiss_index' ada. Error: {e}")
        st.stop()

def initialize_conversation_chain(_llm, _vector_store):
    """
    Membuat chain percakapan dasar.
    """
    retriever = _vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer')

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=_llm,
        retriever=retriever,
        memory=memory,
    )
    
    return conversation_chain


# --- Antarmuka Streamlit (Tidak ada perubahan di bagian ini) ---
st.title("⚡ Chatbot Perjanjian Kerja Bersama (PKB) PT PLN")
st.markdown("""
Selamat datang! Saya adalah asisten AI yang ditenagai oleh **Llama 4 (via OpenRouter)** dan dilatih khusus mengenai dokumen **PKB PT PLN Periode 2025-2027**.
Silakan ajukan pertanyaan Anda terkait isi dokumen tersebut.
""")

llm, vector_store = load_models_and_vector_store()

if "conversation" not in st.session_state:
    st.session_state.conversation = initialize_conversation_chain(llm, vector_store)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_question = st.chat_input("Tanyakan sesuatu tentang PKB PT PLN...")

if user_question:
    st.session_state.chat_history.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.spinner("Llama sedang berpikir..."):
        try:
            result = st.session_state.conversation({"question": user_question})
            answer = result["answer"]

            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            with st.chat_message("assistant"):
                st.markdown(answer)
        except Exception as e:
            error_message = f"Maaf, terjadi kesalahan saat memproses permintaan Anda: {e}"
            st.error(error_message)
            st.session_state.chat_history.append({"role": "assistant", "content": error_message})
