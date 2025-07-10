import os
import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
# BARU: Impor modul tambahan untuk RAG yang lebih canggih
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.retrievers import MultiQueryRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers import ContextualCompressionRetriever

# --- Konfigurasi Awal ---
st.set_page_config(page_title="Chatbot PKB PLN", page_icon="⚡", layout="wide")

# Muat GOOGLE_API_KEY dari secrets Streamlit
if 'GOOGLE_API_KEY' not in os.environ:
    try:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    except Exception as e:
        st.error("Harap atur GOOGLE_API_KEY Anda di Streamlit secrets. Error: " + str(e))
        st.stop()

# --- Template Prompt Kustom ---
# BARU: Kita membuat template instruksi yang lebih spesifik untuk AI
custom_prompt_template = """
Gunakan potongan informasi berikut untuk menjawab pertanyaan pengguna.
Anda adalah Asisten AI yang ahli tentang Perjanjian Kerja Bersama (PKB) PT PLN.
Jika Anda tidak tahu jawabannya berdasarkan konteks yang diberikan, katakan saja bahwa Anda tidak tahu, jangan mencoba mengarang jawaban.
Selalu jawab dalam Bahasa Indonesia yang baik dan jelas.

Konteks: {context}

Pertanyaan: {question}

Jawaban yang membantu:"""

CUSTOM_QUESTION_PROMPT = PromptTemplate.from_template(custom_prompt_template)


# --- Fungsi-Fungsi Inti dengan Cache ---

@st.cache_resource
def load_models_and_vector_store():
    """
    Memuat model LLM, embeddings, dan vector store dari file lokal.
    Menggunakan cache untuk performa.
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
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1, convert_system_message_to_human=True)
        
        # Inisialisasi LLM untuk retriever
        llm_retriever = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

        return llm, llm_retriever, vector_store
    except Exception as e:
        st.error(f"Gagal memuat model atau vector store. Pastikan folder 'pkb_pln_faiss_index' ada. Error: {e}")
        st.stop()

def initialize_advanced_conversation_chain(_llm, _llm_retriever, _vector_store):
    """
    BARU: Membuat chain percakapan yang lebih canggih dengan MultiQuery dan Contextual Compression.
    """
    # 1. Base Retriever: Retriever dasar dari vector store
    base_retriever = _vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 7})

    # 2. MultiQuery Retriever: Menghasilkan beberapa variasi pertanyaan untuk pencarian yang lebih baik
    multiquery_retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever, llm=_llm_retriever
    )

    # 3. Contextual Compression: Menyaring hasil pencarian agar lebih relevan
    compressor = LLMChainExtractor.from_llm(_llm)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=multiquery_retriever
    )
    
    # 4. Inisialisasi memori untuk menyimpan riwayat percakapan
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer')

    # 5. Membuat ConversationalRetrievalChain dengan prompt kustom
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=_llm,
        retriever=compression_retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": CUSTOM_QUESTION_PROMPT},
        verbose=True
    )
    
    return conversation_chain


# --- Antarmuka Streamlit ---
st.title("⚡ Chatbot Perjanjian Kerja Bersama (PKB) PT PLN")
st.markdown("""
Selamat datang! Saya adalah asisten AI yang dilatih khusus mengenai dokumen **PKB PT PLN Periode 2025-2027**.
Silakan ajukan pertanyaan Anda terkait isi dokumen tersebut.
""")

# Muat model dan vector store
llm, llm_retriever, vector_store = load_models_and_vector_store()

# Inisialisasi state untuk menyimpan chain dan riwayat chat
if "conversation" not in st.session_state:
    st.session_state.conversation = initialize_advanced_conversation_chain(llm, llm_retriever, vector_store)
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
    with st.spinner("Menganalisis dokumen dan mencari jawaban..."):
        try:
            result = st.session_state.conversation({"question": user_question})
            answer = result["answer"]
            source_documents = result.get("source_documents", [])

            # Tambahkan jawaban chatbot ke riwayat dan tampilkan
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            with st.chat_message("assistant"):
                st.markdown(answer)
                # BARU: Tampilkan sumber dokumen untuk transparansi
                if source_documents:
                    with st.expander("Lihat Sumber Jawaban"):
                        for doc in source_documents:
                            st.markdown(f"> {doc.page_content}", help=f"Sumber: Halaman {doc.metadata.get('page', 'N/A')}")
        except Exception as e:
            error_message = f"Maaf, terjadi kesalahan saat memproses permintaan Anda: {e}"
            st.error(error_message)
            st.session_state.chat_history.append({"role": "assistant", "content": error_message})
