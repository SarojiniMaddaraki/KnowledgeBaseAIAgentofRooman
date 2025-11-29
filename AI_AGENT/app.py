import streamlit as st
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import google.generativeai as genai

# ========== CONFIGURATION ==========
GEMINI_API_KEY = "AIzaSyDfaZrIBQPiQX3k1LIULsru1no_FRh3XHA"
PINECONE_API_KEY = "pcsk_58nZ62_UMekrnN77cyQzCL5Nm8R2dqxgpHATAKPyzpeCPeybqhYhKmUs6auMihQSKC239f"
INDEX_NAME = "rooman-kb"

# Rooman Brand Colors
ROOMAN_PRIMARY = "#FF6B35"
ROOMAN_SECONDARY = "#004E89"
ROOMAN_ACCENT = "#1A659E"
ROOMAN_DARK = "#2D3142"

# Configure APIs
genai.configure(api_key=GEMINI_API_KEY)

# ========== PAGE CONFIG ==========
st.set_page_config(
    page_title="Rooman AI Agent",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== INITIALIZE ==========
@st.cache_resource
def load_embedding_model():
    """Load sentence transformer model"""
    return SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

@st.cache_resource
def init_pinecone():
    """Initialize Pinecone connection"""
    pc = Pinecone(api_key=PINECONE_API_KEY)
    return pc.Index(INDEX_NAME)

embedding_model = load_embedding_model()
pinecone_index = init_pinecone()
gemini_model = genai.GenerativeModel('gemini-2.5-flash')

# ========== RAG PIPELINE ==========
def rag_answer(question):
    """Complete RAG pipeline with Pinecone"""
    try:
        # Step 1: Create embedding
        query_embedding = embedding_model.encode(question).tolist()
        
        # Step 2: Search Pinecone
        search_results = pinecone_index.query(
            vector=query_embedding,
            top_k=5,
            include_metadata=True
        )
        
        # Step 3: Retrieve chunks
        retrieved_chunks = []
        sources = []
        
        for match in search_results['matches']:
            if 'metadata' in match and 'text' in match['metadata']:
                retrieved_chunks.append(match['metadata']['text'])
            if 'metadata' in match and 'source' in match['metadata']:
                source = match['metadata']['source']
                if source not in sources:
                    sources.append(source)
        
        if not retrieved_chunks:
            return {
                'answer': "❌ No relevant information found in the knowledge base.",
                'context': "",
                'num_chunks': 0,
                'sources': []
            }
        
        context = "\n\n".join(retrieved_chunks)
        
        # Step 4: Generate answer with Gemini
        prompt = f"""You are a helpful AI assistant for Rooman Technologies Knowledge Base.

Context from knowledge base:
{context}

Question: {question}

Instructions:
- Provide detailed, accurate answers based ONLY on the context
- Use bullet points for clarity
- Be professional and encouraging
- Mention specific courses, programs, or benefits when relevant"""
        
        response = gemini_model.generate_content(prompt)
        
        return {
            'answer': response.text,
            'context': context,
            'num_chunks': len(retrieved_chunks),
            'sources': sources
        }
        
    except Exception as e:
        return {
            'answer': f"❌ Error: {str(e)}",
            'context': "",
            'num_chunks': 0,
            'sources': []
        }

# ========== SESSION STATE ==========
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'show_debug' not in st.session_state:
    st.session_state.show_debug = False
if 'user_email' not in st.session_state:
    st.session_state.user_email = ""

# ========== CUSTOM CSS ==========
st.markdown(f"""
    <style>
    /* Remove default padding */
    .main .block-container {{
        padding-top: 2rem;
        padding-bottom: 0rem;
        max-width: 100%;
    }}
    
    /* Main Header */
    .main-header {{
        background: linear-gradient(135deg, {ROOMAN_PRIMARY} 0%, {ROOMAN_SECONDARY} 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }}
    
    .main-header h1 {{
        font-size: 2.5rem;
        margin: 0;
        font-weight: 800;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }}
    
    .main-header p {{
        font-size: 1.2rem;
        margin-top: 0.5rem;
        opacity: 0.95;
    }}
    
    /* Login Container - Fixed Height */
    .login-container {{
        display: flex;
        align-items: center;
        justify-content: center;
        min-height: 60vh;
        padding: 1rem 0;
    }}
    
    .login-box {{
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.15);
        width: 100%;
        max-width: 450px;
    }}
    
    .login-box h3 {{
        color: {ROOMAN_PRIMARY};
        text-align: center;
        margin-bottom: 1rem;
        font-size: 1.3rem;
    }}
    
    /* Source Tags */
    .source-tag {{
        display: inline-block;
        background: linear-gradient(135deg, {ROOMAN_ACCENT} 0%, {ROOMAN_SECONDARY} 100%);
        color: white;
        padding: 0.4rem 0.9rem;
        border-radius: 20px;
        margin: 0.3rem;
        font-size: 0.9rem;
        font-weight: 500;
    }}
    
    /* Buttons */
    .stButton>button {{
        background: linear-gradient(135deg, {ROOMAN_PRIMARY} 0%, {ROOMAN_SECONDARY} 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
    }}
    
    .stButton>button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 5px 20px rgba(255, 107, 53, 0.4);
    }}
    
    /* Metrics */
    .metric-box {{
        background: linear-gradient(135deg, {ROOMAN_PRIMARY} 0%, {ROOMAN_SECONDARY} 100%);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }}
    
    /* Info Box */
    .info-box {{
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        border-left: 4px solid {ROOMAN_PRIMARY};
        padding: 1.2rem;
        border-radius: 10px;
        margin: 1rem 0;
    }}
    
    .info-box h3 {{
        color: {ROOMAN_SECONDARY};
        margin-top: 0;
        font-size: 1.2rem;
    }}
    
    .info-box ul {{
        margin: 0.5rem 0;
        padding-left: 1.5rem;
    }}
    
    /* Hide Streamlit branding */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    </style>
""", unsafe_allow_html=True)

# ========== LOGIN FORM ==========
def show_login_form():
    # Header
    st.markdown("""
        <div class="main-header">
            <h1>🎓 Rooman AI Agent</h1>
            <p>Your Intelligent Knowledge Assistant</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Centered login container
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown('<div class="login-box">', unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["🔐 Login", "📝 Sign Up"])
        
        with tab1:
            st.markdown("### Welcome Back!")
            email = st.text_input("Email Address", key="login_email", placeholder="your.email@example.com")
            password = st.text_input("Password", type="password", key="login_password", placeholder="Enter password")
            
            if st.button("🚀 Login to Dashboard", use_container_width=True, type="primary"):
                if email and password and len(password) >= 6:
                    st.session_state.authenticated = True
                    st.session_state.user_email = email
                    st.success("✅ Login successful!")
                    st.rerun()
                else:
                    st.error("❌ Invalid credentials. Password must be 6+ characters.")
        
        with tab2:
            st.markdown("### Create New Account")
            new_email = st.text_input("Email Address", key="signup_email", placeholder="your.email@example.com")
            new_password = st.text_input("Password", type="password", key="signup_password", placeholder="Min 6 characters")
            confirm_password = st.text_input("Confirm Password", type="password", key="confirm_password", placeholder="Re-enter password")
            
            if st.button("✨ Create Account", use_container_width=True, type="primary"):
                if new_email and new_password and confirm_password:
                    if new_password == confirm_password:
                        if len(new_password) >= 6:
                            st.session_state.authenticated = True
                            st.session_state.user_email = new_email
                            st.success("✅ Account created!")
                            st.rerun()
                        else:
                            st.error("❌ Password must be 6+ characters")
                    else:
                        st.error("❌ Passwords don't match")
                else:
                    st.error("⚠️ Please fill all fields")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Info section below login
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
            <div class="info-box">
                <h3>🎯 What You Can Do:</h3>
                <ul>
                    <li>Ask about Rooman courses and programs</li>
                    <li>Get information on PMKVY initiatives</li>
                    <li>Learn about job-guaranteed programs</li>
                    <li>Explore certifications and career paths</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

# ========== MAIN DASHBOARD ==========
def show_dashboard():
    # Header
    st.markdown("""
        <div class="main-header">
            <h1>🎓 Rooman Knowledge Base AI Agent</h1>
            <p>Ask anything about courses, certifications, PMKVY & career guidance</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 👤 User Profile")
        st.markdown(f"**📧** {st.session_state.user_email}")
        
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.messages = []
            st.session_state.user_email = ""
            st.rerun()
        
        st.markdown("---")
        
        st.markdown("### ⚙️ Settings")
        st.session_state.show_debug = st.checkbox("🔍 Debug Mode", value=False)
        
        st.markdown("---")
        
        st.markdown("### 📊 Statistics")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f'<div class="metric-box"><h2>{len(st.session_state.messages)}</h2><p>Messages</p></div>', unsafe_allow_html=True)
        with col2:
            queries = len([m for m in st.session_state.messages if m["role"] == "user"])
            st.markdown(f'<div class="metric-box"><h2>{queries}</h2><p>Queries</p></div>', unsafe_allow_html=True)
        
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        
        st.markdown("---")
        
        st.markdown("### 💡 Quick Questions")
        examples = [
            "What courses does Rooman offer?",
            "Tell me about PMKVY programs",
            "Job-guaranteed programs?",
            "Data Science syllabus?",
            "How to enroll?"
        ]
        
        for i, eq in enumerate(examples):
            if st.button(f"📌 {eq}", key=f"ex_{i}", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": eq})
                with st.spinner("🤔 Thinking..."):
                    result = rag_answer(eq)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result['answer'],
                        "sources": result['sources']
                    })
                st.rerun()
        
        st.markdown("---")
        st.success("✅ Pinecone Connected")
        st.success("✅ Gemini AI Active")
        st.success("✅ Embeddings Ready")
    
    # Welcome message
    if len(st.session_state.messages) == 0:
        st.markdown("""
            <div class="info-box">
                <h3>👋 Welcome to Rooman Knowledge Base!</h3>
                <p><strong>I can help you with:</strong></p>
                <ul>
                    <li>🎯 Course information and detailed syllabus</li>
                    <li>📜 Certifications and training programs</li>
                    <li>💼 Job-guaranteed programs with placement</li>
                    <li>🏆 PMKVY government initiatives</li>
                    <li>🚀 Career guidance and learning paths</li>
                </ul>
                <p><strong>Ask me anything or use quick questions from the sidebar!</strong></p>
            </div>
        """, unsafe_allow_html=True)
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            if message["role"] == "assistant" and "sources" in message and message["sources"]:
                st.markdown("**📚 Sources:**")
                for source in message["sources"]:
                    st.markdown(f'<span class="source-tag">{source}</span>', unsafe_allow_html=True)
    
    # Chat input
    if question := st.chat_input("💭 Ask your question here..."):
        st.session_state.messages.append({"role": "user", "content": question})
        
        with st.chat_message("user"):
            st.markdown(question)
        
        with st.chat_message("assistant"):
            with st.spinner("🤔 Searching knowledge base..."):
                result = rag_answer(question)
                st.markdown(result['answer'])
                
                if result['sources']:
                    st.markdown("**📚 Sources:**")
                    for source in result['sources']:
                        st.markdown(f'<span class="source-tag">{source}</span>', unsafe_allow_html=True)
                
                if st.session_state.show_debug:
                    with st.expander("🔍 Debug Information"):
                        st.metric("Chunks Retrieved", result['num_chunks'])
                        st.text_area("Context", result['context'], height=150)
        
        st.session_state.messages.append({
            "role": "assistant",
            "content": result['answer'],
            "sources": result['sources']
        })

# ========== MAIN LOGIC ==========
if not st.session_state.authenticated:
    show_login_form()
else:
    show_dashboard()
