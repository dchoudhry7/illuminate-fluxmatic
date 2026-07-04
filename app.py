import os
import sys
import time
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import streamlit as st
import pypdf
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE_DIR, "chroma_db")
DATA_DIR = os.path.join(BASE_DIR, "data")
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")

st.set_page_config(
    page_title="Illuminate - Fluxmatic Lighting AI",
    page_icon=":material/lightbulb:",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown(
    """
    <style>
        h1 a {
            text-decoration: none !important;
            color: inherit !important;
        }
    </style>
    <script>
        var forceSelfTarget = function() {
            var links = window.parent.document.querySelectorAll('h1 a');
            for (var i = 0; i < links.length; i++) {
                if (links[i].getAttribute('target') !== '_self') {
                    links[i].setAttribute('target', '_self');
                }
            }
        };
        setInterval(forceSelfTarget, 200);
    </script>
    """,
    unsafe_allow_html=True
)

@st.cache_resource
def get_db_and_llm():
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
    except Exception as e:
        embeddings = None
    vector_db = None
    if os.path.exists(DB_DIR):
        try:
            vector_db = Chroma(
                persist_directory=DB_DIR,
                embedding_function=embeddings
            )
        except Exception as e:
            pass
    return vector_db, None, embeddings

@st.cache_resource
def get_llm():
    llm = None
    api_key = os.getenv("GROQ_API_KEY")
    if api_key:
        try:
            llm = ChatGroq(
                temperature=0.25,
                model_name="llama-3.1-8b-instant",
                api_key=api_key
            )
        except Exception as e:
            pass
    return llm

db, _, embeddings = get_db_and_llm()
llm = get_llm()

def render_chat_message(role, text):
    if role == "user":
        st.markdown(
            f"""
            <div style="display: flex; justify-content: flex-end; margin-bottom: 12px;">
                <div style="
                    background-color: rgba(128, 128, 128, 0.08); 
                    border: 1px solid rgba(128, 128, 128, 0.15); 
                    border-radius: 12px; 
                    padding: 10px 14px; 
                    max-width: 80%;
                ">
                    {text}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div style="display: flex; justify-content: flex-start; margin-bottom: 12px;">
                <div style="
                    background-color: rgba(96, 165, 250, 0.1); 
                    border: 1px solid rgba(96, 165, 250, 0.25); 
                    border-radius: 12px; 
                    padding: 10px 14px; 
                    max-width: 80%;
                ">
                    {text}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

overview_mode = st.query_params.get("overview", "false") == "true"

if overview_mode:
    st.markdown("# [:material/lightbulb: illuminate](/?overview=false)")
    st.caption("Fluxmatic Architectural Lighting Consultant")
    st.write("---")
    
    st.markdown("### Company Profile")
    st.write(
        "Fluxmatic Solution LLP is an Indian Limited Liability Partnership (LLP) incorporated on July 7, 2022. "
        "The company specializes in importing and supplying high-quality architectural, linear, outdoor, and landscape "
        "lighting fixtures from world-renowned international brands."
    )
    
    st.markdown("### Value Proposition")
    st.write(
        "At Fluxmatic, we help architects, designers, and project builders bridge the gap between abstract design "
        "concepts and technical lighting reality. We curate a premium selection of luminaires focused on "
        "high performance, visual comfort, low glare, and durability."
    )
    
    st.markdown("### Contact & Location")
    st.write("**Operations Office Address**:")
    st.write(
        "B-46, 2nd Floor, DDA Shed,  \n"
        "Okhla Industrial Area, Phase-2,  \n"
        "New Delhi - 110020, India."
    )
    
    col_l, col_k = st.columns(2)
    with col_l:
        st.markdown("**Love** (Co-founder / Partner)")
        st.markdown(":material/phone: Phone: +91 9643685518  \n:material/mail: Email: love@fluxmatic.in")
    with col_k:
        st.markdown("**Kush** (Co-founder / Partner)")
        st.markdown(":material/phone: Phone: +91 9999683936  \n:material/mail: Email: kush@fluxmatic.in")
        
    st.markdown("### Partner Brands Portfolio")
    st.write("- **XAL (Austria)**: Premium linear profiles and architectural ceiling lighting systems.")
    st.write("- **Wever & Ducré (Belgium)**: Trendy and cozy decorative interior fixtures.")
    st.write("- **NEKO (Switzerland)**: Swiss-engineered modular downlights and low-glare spots.")
    st.write("- **Wästberg (Sweden)**: Circadian-rhythm task lights and designer workspace lamps.")
    st.write("- **Unilamp (Thailand)**: Waterproof outdoor facade fixtures and landscape projectors.")
    
    st.write("---")
    if st.button("Back to Chat", icon=":material/chat:", use_container_width=True):
        st.query_params["overview"] = "false"
        st.rerun()

else:
    st.markdown("# [:material/lightbulb: illuminate](/?overview=true)")
    st.caption("Fluxmatic Architectural Lighting Consultant")
    st.write("---")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    if not st.session_state.messages:
        st.write("### Try asking:")
        col1, col2 = st.columns(2)
        p1 = col1.button("Fluxmatic Brand Portfolio", icon=":material/menu_book:", use_container_width=True)
        p2 = col2.button("Office Address & Contacts", icon=":material/business:", use_container_width=True)
        p3 = col1.button("Switzerland's NEKO Details", icon=":material/settings:", use_container_width=True)
        p4 = col2.button("Sweden's Wästberg Lamps", icon=":material/lightbulb:", use_container_width=True)
        
        prompt = None
        if p1:
            prompt = "Tell me about Fluxmatic's lighting brand portfolio and what products they offer."
        elif p2:
            prompt = "What is Fluxmatic Solution LLP's official registered office address, phone numbers, and email contacts?"
        elif p3:
            prompt = "Detail Switzerland's NEKO downlights and modular spotlights. What makes their engineering special?"
        elif p4:
            prompt = "What are Sweden's Wästberg lighting products and who are the famous designers they collaborate with?"
            
        if prompt:
            st.session_state.starter_prompt = prompt
            st.rerun()
            
    for msg in st.session_state.messages:
        render_chat_message(msg["role"], msg["content"])
        
    user_query = None
    if "starter_prompt" in st.session_state and st.session_state.starter_prompt:
        user_query = st.session_state.starter_prompt
        del st.session_state.starter_prompt
        
    chat_input_val = st.chat_input("Talk to Illuminate...")
    if chat_input_val:
        user_query = chat_input_val
        
    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        render_chat_message("user", user_query)
        
        if not llm:
            st.error("Groq LLM is not configured. Please set the GROQ_API_KEY environment variable in your .env file.")
        else:
            context = ""
            if db is not None:
                try:
                    retrieved_docs = db.similarity_search(user_query, k=3)
                    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
                except Exception as e:
                    pass
                    
            system_prompt = (
                "You are Illuminate, an expert lighting design consultant and customer relations AI for Fluxmatic Solution LLP. "
                "Your objective is to answer questions about Fluxmatic, its address, contacts (Love at love@fluxmatic.in, Kush at kush@fluxmatic.in), value proposition, and its architectural lighting brand partners: "
                "1. XAL (Austria) - Architectural linear profile structures and workplace track channels.\n"
                "2. Wever & Ducré (Belgium) - Cozy, trendy, aesthetic retail/home lighting.\n"
                "3. NEKO (Switzerland) - Precision LED downlights with Switzerland modular technology and low glare.\n"
                "4. Wästberg / Wastberg (Sweden) - Award-winning designer table, task, and desk lamps for human-centric well-being.\n"
                "5. Unilamp (Thailand) - Waterproof IP65-IP67 landscape projectors, inground uplights, and facade spotlights.\n\n"
                "Answer the query accurately based on the provided context. If the context does not contain the answer, "
                "use your general knowledge to provide a helpful, creative, and professional response. Maintain a premium "
                "brand tone. Highlight how Fluxmatic bridges the gap between design concepts and built lighting reality.\n\n"
                "Format your reply using professional markdown lists and headers."
            )
            
            messages = [
                ("system", system_prompt),
                ("human", f"Context from documentation:\n{context}\n\nQuestion: {user_query}")
            ]
            
            try:
                with st.spinner("Thinking..."):
                    response = llm.invoke(messages)
                    response_text = response.content
                    
                words = response_text.split(" ")
                current_text = ""
                placeholder = st.empty()
                for word in words:
                    current_text += word + " "
                    placeholder.markdown(
                        f"""
                        <div style="display: flex; justify-content: flex-start; margin-bottom: 12px;">
                            <div style="
                                background-color: rgba(96, 165, 250, 0.1); 
                                border: 1px solid rgba(96, 165, 250, 0.25); 
                                border-radius: 12px; 
                                padding: 10px 14px; 
                                max-width: 80%;
                            ">
                                {current_text}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    time.sleep(0.03)
                    
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response_text
                })
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")
