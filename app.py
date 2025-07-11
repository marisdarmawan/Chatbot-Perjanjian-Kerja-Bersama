import os
import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# --- Konfigurasi Awal ---
st.set_page_config(page_title="Chatbot PKB PLN", page_icon="⚡", layout="wide")

# Muat GOOGLE_API_KEY dari secrets Streamlit
if 'GOOGLE_API_KEY' not in os.environ:
    try:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    except Exception as e:
        st.error("Harap atur GOOGLE_API_KEY Anda di Streamlit secrets. Error: " + str(e))
        st.stop()


# --- Fungsi-Fungsi Inti dengan Cache ---

@st.cache_resource
def load_models_and_vector_store():
    """
    Memuat model LLM, embeddings, dan vector store dari file lokal.
    """
    try:
        # Inisialisasi model embeddings
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

        # Muat vector store dari file lokal
        vector_store = FAISS.load_local(
            "pkb_pln_faiss_index",
            embeddings,
            allow_dangerous_deserialization=True
        )

        # Inisialisasi model Chat utama
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2, convert_system_message_to_human=True)
        
        return llm, vector_store
    except Exception as e:
        st.error(f"Gagal memuat model atau vector store. Pastikan folder 'pkb_pln_faiss_index' ada. Error: {e}")
        st.stop()

def initialize_conversation_chain(_llm, _vector_store):
    """
    DISEDERHANAKAN: Membuat chain percakapan dasar tanpa retriever canggih.
    """
    # 1. Membuat retriever dasar langsung dari vector store
    retriever = _vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    
    # 2. Inisialisasi memori untuk menyimpan riwayat percakapan
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer')

    # 3. Membuat ConversationalRetrievalChain standar
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=_llm,
        retriever=retriever,
        memory=memory,
    )
    
    return conversation_chain


# --- Antarmuka Streamlit ---
st.title("⚡ Chatbot Perjanjian Kerja Bersama (PKB) PT PLN")
st.markdown("""
Selamat datang! Saya adalah asisten AI yang dilatih khusus mengenai dokumen **PKB PT PLN Periode 2025-2027**.
Silakan ajukan pertanyaan Anda terkait isi dokumen tersebut.
Contoh : Berapa lama cuti melahirkan bagi pegawai wanita? 
""")

# Muat model dan vector store
llm, vector_store = load_models_and_vector_store()

# Inisialisasi state untuk menyimpan chain dan riwayat chat
if "conversation" not in st.session_state:
    # DIUBAH: Memanggil fungsi yang sudah disederhanakan
    st.session_state.conversation = initialize_conversation_chain(llm, vector_store)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Menampilkan riwayat chat
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input dari pengguna
user_question = st.chat_input("Tanyakan sesuatu tentang PKB PT PLN...")

if user_question:
    # Tambahkan pertanyaan pengguna ke riwayat dan tampilkan
    st.session_state.chat_history.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    # Proses pertanyaan dan dapatkan jawaban
    with st.spinner("Mencari jawaban..."):
        try:
            result = st.session_state.conversation({"question": user_question})
            answer = result["answer"]

            # Tambahkan jawaban chatbot ke riwayat dan tampilkan
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            with st.chat_message("assistant"):
                st.markdown(answer)
        except Exception as e:
            error_message = f"Maaf, terjadi kesalahan saat memproses permintaan Anda: {e}"
            st.error(error_message)
            st.session_state.chat_history.append({"role": "assistant", "content": error_message})
