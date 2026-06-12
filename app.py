import os
from groq import Groq
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings, HuggingFaceEmbeddings
from langchain_community.llms import Ollama
import streamlit as st

# ── Config ────────────────────────────────────────────────────────
VECTORSTORE_DIR = "vectorstore"
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "llama3.1"

SYSTEM_PROMPT = """You are Simone de Beauvoir, the French existentialist philosopher and feminist thinker (1908–1986).
You are speaking in the first person, drawing on your own philosophical writings and ideas.

Rules you must follow:
- Respond only based on the source passages provided below. Do not invent positions you have not argued.
- If the passages do not contain enough to answer well, say so honestly — acknowledge the limit rather than speculate.
- Speak in a measured, intellectual tone — thoughtful, direct, occasionally passionate, never casual.
- Refer to your own works naturally (e.g. "In The Ethics of Ambiguity, I argued...").
- Do not break character. You are de Beauvoir, not an AI assistant.
- End responses with a question or provocation that invites the user to think further.

Source passages from your writings:
{context}

User's question: {question}

Your response:"""

# ── Load vectorstore ──────────────────────────────────────────────
@st.cache_resource
def load_vectorstore():
    if os.environ.get("STREAMLIT_CLOUD"):
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    else:
        embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    return Chroma(
        persist_directory=VECTORSTORE_DIR,
        embedding_function=embeddings
    )

# ── Retrieve relevant chunks ──────────────────────────────────────
def retrieve(query, vectorstore, k=4):
    results = vectorstore.similarity_search_with_score(query, k=k)
    return results

# ── Generate response ─────────────────────────────────────────────
def generate(question, context):
    prompt = SYSTEM_PROMPT.format(context=context, question=question)

    groq_key = os.environ.get("GROQ_API_KEY")

    if groq_key:
        # Cloud: use Groq
        client = Groq(api_key=groq_key)
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    else:
        # Local: use Ollama
        llm = Ollama(model=CHAT_MODEL)
        return llm.invoke(prompt)

# ── Build context string from chunks ─────────────────────────────
def build_context(results):
    context_parts = []
    for doc, score in results:
        context_parts.append(f"[From: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}")
    return "\n\n---\n\n".join(context_parts)

# ── Styling ───────────────────────────────────────────────────────
def apply_styling():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;1,400&family=Source+Sans+3:wght@300;400;600&display=swap');
        @import url('https://fonts.googleapis.com/icon?family=Material+Icons');
        
        [data-testid="stIconMaterial"] {
        font-family: 'Material Icons' !important;
        font-size: 18px !important;
        }
    
        .stApp {
            background-color: #F5F0E8;
        }
        [data-testid="stAppViewContainer"] {
            background-color: #F5F0E8;
        }
        [data-testid="stHeader"] {
            background-color: #F5F0E8;
        }
        h1 {
            font-family: 'Playfair Display', serif !important;
            color: #2C2416 !important;
            font-weight: 600 !important;
        }
        body, p, div, span, label, .stMarkdown {
            font-family: 'Source Sans 3', sans-serif !important;
            color: #2C2416 !important;
        }
        .stCaption {
            font-family: 'Source Sans 3', sans-serif !important;
            color: #7A6A52 !important;
            font-style: italic;
        }
        .stChatInput input {
            background-color: #EDE8DE !important;
            border: 1px solid #C4B89A !important;
            font-family: 'Source Sans 3', sans-serif !important;
            color: #2C2416 !important;
        }
        .stChatMessage {
            background-color: #EDE8DE !important;
            border: 1px solid #C4B89A !important;
            border-radius: 8px !important;
        }
        [data-testid="stSidebar"] {
            background-color: #EDE8DE !important;
            border-right: 1px solid #C4B89A !important;
        }
        .stAlert {
            background-color: #E8E0CC !important;
            border: 1px solid #C4B89A !important;
            color: #2C2416 !important;
        }
       .streamlit-expanderHeader {
            font-family: 'Source Sans 3', sans-serif !important;
            color: #2C2416 !important;
            padding-left: 24px !important;
        }

        [data-testid="stExpander"] summary {
            padding-left: 24px !important;
        }
    
        </style>
    """, unsafe_allow_html=True)

# ── Streamlit UI ──────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="In Dialogue with Beauvoir",
        page_icon="✒️",
        layout="wide"
    )
    apply_styling()

    st.title("✒️ In Dialogue with Beauvoir")
    st.caption("A RAG-powered philosophical interlocutor grounded in the writings of Simone de Beauvoir")

    st.warning(
        "⚠️ This is an AI-generated interpretation grounded in source texts — not statements "
        "de Beauvoir herself made. Always consult the original works.",
        icon="📜"
    )

    vectorstore = load_vectorstore()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    question = st.chat_input("Ask Simone de Beauvoir a question...")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.spinner("Consulting the texts..."):
            results = retrieve(question, vectorstore)
            st.session_state.last_sources = results
            context = build_context(results)
            response = generate(question, context)

        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant", avatar="✒️"):
            st.write(response)

    with st.sidebar:
        st.subheader("📄 Source passages retrieved")
        st.caption("These are the passages the response was grounded in.")
        for i, (doc, score) in enumerate(results):
            with st.expander(f"Passage {i+1} · {doc.metadata.get('source', 'unknown')}"):
                st.write(doc.page_content)
                st.caption(f"Relevance score: {round(score, 3)}")

if __name__ == "__main__":
    main()