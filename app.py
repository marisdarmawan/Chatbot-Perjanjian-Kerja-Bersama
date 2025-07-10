import os
import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# --- Konfigurasi Awal ---
# Atur GOOGLE_API_KEY Anda.
# Sebaiknya gunakan st.secrets untuk keamanan saat deploy.
# Untuk pengembangan lokal, bisa set langsung.
# Contoh: os.environ["GOOGLE_API_KEY"] = "AIza..."
# Pastikan Anda telah mengatur GOOGLE_API_KEY di environment variables atau secrets Streamlit
if 'GOOGLE_API_KEY' not in os.environ:
    try:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    except:
        st.error("Harap atur GOOGLE_API_KEY Anda di Streamlit secrets.")
        st.stop()


# --- Fungsi-Fungsi Inti ---

@st.cache_resource
def load_llm_and_retriever():
    """
    Memuat model LLM dan retriever dari vector store yang sudah ada.
    Menggunakan cache untuk performa.
    """
    # Inisialisasi model embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Muat vector store dari file lokal
    try:
        vector_store = FAISS.load_local(
            "pkb_pln_faiss_index",
            embeddings,
            allow_dangerous_deserialization=True # Diperlukan untuk FAISS versi baru
        )
    except Exception as e:
        st.error(f"Gagal memuat indeks FAISS. Pastikan folder 'pkb_pln_faiss_index' ada di direktori yang sama. Error: {e}")
        st.stop()

    # Buat retriever dari vector store
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    # Inisialisasi model Chat
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)

    return llm, retriever

def initialize_conversation_chain(_llm, _retriever):
    """
    Membuat chain percakapan yang menyimpan riwayat obrolan.
    """
    # Inisialisasi memori untuk menyimpan riwayat percakapan
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

    # Buat chain percakapan
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=_llm,
        retriever=_retriever,
        memory=memory,
        verbose=True # Tampilkan proses di terminal
    )
    return conversation_chain


# --- Antarmuka Streamlit ---

st.set_page_config(page_title="Chatbot PKB PLN", page_icon="⚡")

st.title("⚡ Chatbot Perjanjian Kerja Bersama (PKB) PT PLN")
st.markdown("""
Selamat datang! Saya adalah asisten AI yang dilatih khusus mengenai dokumen **PKB PT PLN Periode 2025-2027**.
Silakan ajukan pertanyaan Anda terkait isi dokumen tersebut.
Contoh: *'Berapa lama cuti melahirkan untuk pegawai wanita?'* atau *'Apa saja yang termasuk dalam kompensasi tetap?'*
""")

# Muat model dan retriever
llm, retriever = load_llm_and_retriever()

# Inisialisasi state untuk menyimpan chain dan riwayat chat
if "conversation" not in st.session_state:
    st.session_state.conversation = initialize_conversation_chain(llm, retriever)
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
    with st.spinner("Sedang mencari jawaban..."):
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
