import streamlit as st
import random
import json
import re
from datetime import datetime

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="AI Mock Test Generator",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown("""
<style>
    .block-container { padding-top: 1rem; padding-bottom: 2rem; }
    .question-card {
        background: rgba(30, 136, 229, 0.05);
        padding: 25px;
        border-radius: 12px;
        border-left: 8px solid #1E88E5;
        margin-bottom: 20px;
    }
    .stButton > button {
        width: 100%;
        height: 50px;
        font-size: 18px;
        font-weight: bold;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================================
# SESSION STATE
# ==========================================================

if "quiz_data" not in st.session_state:
    st.session_state.quiz_data = []
if "current_q_idx" not in st.session_state:
    st.session_state.current_q_idx = 0
if "quiz_submitted" not in st.session_state:
    st.session_state.quiz_submitted = False
if "user_answers" not in st.session_state:
    st.session_state.user_answers = {}

# ==========================================================
# BUILT-IN KNOWLEDGE BASE
# ==========================================================

study_material = """
DATA STRUCTURES: Arrays, Searching, Sorting, Binary Search, Sliding Window, Two Pointer, Prefix Sum, Strings, Linked List, Stack, Queue, Trees, BST, Graphs, DFS, BFS, Dynamic Programming.
DBMS: Normalization, Transactions, SQL, Joins, Indexing.
Operating System: CPU Scheduling, Deadlock, Paging, Memory Management.
QUANTITATIVE APTITUDE: Percentage, Profit and Loss, Simple Interest, Compound Interest, Average, Ratio and Proportion, Probability, Permutation, Combination, Number System, Time and Work, Time Speed Distance.
Logical Reasoning: Blood Relations, Directions, Coding Decoding, Seating Arrangement, Syllogism.
"""

# ==========================================================
# VECTOR DATABASE & RETRIEVER
# ==========================================================

@st.cache_resource
def load_vector_database():
    try:
        splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
        chunks = splitter.split_text(study_material)
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        return FAISS.from_texts(texts=chunks, embedding=embeddings)
    except Exception as e:
        st.error(f"Failed to load vector database: {str(e)}")
        return None

vector_db = load_vector_database()
retriever = vector_db.as_retriever(search_type="mmr", search_kwargs={"k": 6, "fetch_k": 20}) if vector_db else None

# ==========================================================
# PROMPTS & JSON PARSER
# ==========================================================

QUIZ_PROMPT = ChatPromptTemplate.from_template("""
You are a Senior Aptitude Assessment Designer.
Create ONE multiple-choice question for a {company} interview assessment.
Topic: {topic}
Difficulty: {difficulty}
Reference Context: {context}

CRITICAL INSTRUCTIONS FOR EXPLANATION:
- Solve the math problem silently in your head first.
- DO NOT ramble, repeat yourself, or mention the options in the explanation.
- The explanation MUST be divided into an array of clear, concise steps.
- Provide an alternative shortcut method if applicable.

OUTPUT STRICTLY IN JSON FORMAT with this exact schema:
{{
  "question": "The full text of the question",
  "options": ["Option A text", "Option B text", "Option C text", "Option D text"],
  "correct_option": "Exact text of the correct option from the list above",
  "explanation_steps": [
    "1. Calculate the Cost Price (CP): Total weight bought: 100 kg, Cost per kg: $5, Total Cost Price: 100 × $5 = $500",
    "2. Calculate Selling Price for the First Part (80%): ...",
    "3. Calculate Selling Price for the Remaining Part (20%): ...",
    "4. Determine Total Profit Percentage: Total Selling Price: $480 + $90 = $570. Total Profit made: $570 - $500 = $70. Profit percentage equation: ($70 / $500) × 100 = 14%"
  ],
  "shortcut_method": "Alternative Shortcut Method: You can bypass the dollar amount entirely by using a weighted average. 80% of items yield +20% profit. 20% of items yield -10% loss. Net effect: (0.80 × 20) + (0.20 × -10). Final calculation: 16 - 2 = 14%"
}}

Do not include any markdown formatting or text outside the JSON block.
""")

def extract_json(text):
    text = text.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1:
            try:
                return json.loads(text[start:end+1])
            except:
                pass
    return None

# ==========================================================
# QUIZ GENERATION LOGIC
# ==========================================================

def generate_quiz_question(llm, company, topic, difficulty):
    if llm is None:
        return None, "LLM not initialized."
    
    context = "\n".join([doc.page_content for doc in retriever.invoke(topic)]) if retriever else ""
    
    prompt = QUIZ_PROMPT.format(
        company=company, topic=topic, difficulty=difficulty, context=context
    )
    
    try:
        response = llm.invoke(prompt)
        q_data = extract_json(response.content)
        
        if q_data and "question" in q_data and "options" in q_data and "correct_option" in q_data:
            return q_data, None
        return None, "Failed to parse JSON from LLM response."
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "rate_limit_exceeded" in error_msg:
            return None, "Rate limit reached for this model. Please switch models in the sidebar or wait for the limit to reset."
        return None, f"API Error: {error_msg}"

def generate_full_quiz(llm, company, topic, difficulty, count):
    quiz = []
    progress = st.progress(0)
    status = st.empty()
    
    for i in range(count):
        status.info(f"Generating Question {i+1} of {count}...")
        q, error = generate_quiz_question(llm, company, topic, difficulty)
        if q:
            quiz.append(q)
        elif error:
            status.error(error)
            st.stop() # Halt generation if we hit a rate limit
            
        progress.progress((i + 1) / count)
        
    progress.empty()
    status.empty()
    return quiz

# ==========================================================
# SIDEBAR CONFIGURATION
# ==========================================================

st.sidebar.title("⚙ Configuration")

# Model selector to bypass rate limits
model_choice = st.sidebar.selectbox(
    "Select Groq Model", 
    ["llama-3.3-70b-versatile", "llama3-8b-8192", "gemma2-9b-it"],
    help="Switch to llama3-8b or gemma2 if you hit a 429 rate limit error on the 70b model."
)

company = st.sidebar.selectbox("Target Company", ["Google", "Amazon", "Microsoft", "Adobe", "Uber", "Goldman Sachs"])
difficulty = st.sidebar.selectbox("Difficulty", ["Easy", "Medium", "Hard"])
num_questions = st.sidebar.slider("Number of Questions", 1, 10, 5)
topic = st.sidebar.text_input("Enter Topic", placeholder="Arrays, Percentage, DBMS...")

# Initialize LLM based on selection
llm = None
try:
    groq_api_key = st.secrets.get("GROQ_API_KEY", "")
    if groq_api_key:
        llm = ChatGroq(api_key=groq_api_key, model=model_choice, temperature=0.2, max_tokens=4096)
except Exception:
    pass

if st.sidebar.button("🚀 Start New Test"):
    if not topic.strip():
        st.sidebar.warning("Please enter a topic.")
    elif llm is None:
        st.sidebar.error("Groq LLM not connected.")
    else:
        with st.spinner(f"Generating your test using {model_choice}..."):
            st.session_state.quiz_data = generate_full_quiz(llm, company, topic, difficulty, num_questions)
            if st.session_state.quiz_data: # Only start if generation succeeded
                st.session_state.current_q_idx = 0
                st.session_state.quiz_submitted = False
                st.session_state.user_answers = {}
        st.rerun()

# ==========================================================
# MAIN UI ROUTING
# ==========================================================

st.title("🎯 AI Mock Test Generator")
st.caption("Interactive assessment platform powered by RAG + Groq")

if llm is None:
    st.error("❌ Groq LLM not initialized. Please set GROQ_API_KEY in secrets.")
elif not st.session_state.quiz_data:
    st.info("👈 Configure your test in the sidebar and click 'Start New Test' to begin.")
    
elif not st.session_state.quiz_submitted:
    # ==========================================
    # QUIZ TAKING VIEW (ONE BY ONE)
    # ==========================================
    q_idx = st.session_state.current_q_idx
    total_q = len(st.session_state.quiz_data)
    current_q = st.session_state.quiz_data[q_idx]
    
    st.header(f"Question {q_idx + 1} of {total_q}")
    st.progress((q_idx + 1) / total_q)
    
    with st.container():
        st.markdown('<div class="question-card">', unsafe_allow_html=True)
        st.markdown(f"### {current_q['question']}")
        
        default_idx = st.session_state.user_answers.get(q_idx, None)
        options = current_q["options"]
        try:
            default_selection = options.index(default_idx) if default_idx in options else None
        except:
            default_selection = None
            
        selected_option = st.radio(
            "Select your answer:",
            options,
            index=default_selection,
            key=f"radio_{q_idx}"
        )
        
        if selected_option:
            st.session_state.user_answers[q_idx] = selected_option
            
        st.markdown('</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if q_idx > 0:
            if st.button("⬅ Previous"):
                st.session_state.current_q_idx -= 1
                st.rerun()
                
    with col2:
        if q_idx < total_q - 1:
            if st.button("Next ➡"):
                st.session_state.current_q_idx += 1
                st.rerun()
        else:
            if st.button("✅ Submit Test", type="primary"):
                st.session_state.quiz_submitted = True
                st.rerun()

else:
    # ==========================================
    # RESULTS & EXPLANATION VIEW
    # ==========================================
    st.header("🏆 Test Results")
    
    score = 0
    total_q = len(st.session_state.quiz_data)
    
    for idx, q_data in enumerate(st.session_state.quiz_data):
        user_ans = st.session_state.user_answers.get(idx, "No Answer")
        correct_ans = q_data["correct_option"]
        is_correct = (user_ans == correct_ans)
        
        if is_correct:
            score += 1
            border_color = "#28a745"
            status_icon = "✅"
        else:
            border_color = "#dc3545"
            status_icon = "❌"
            
        st.markdown(f"""
        <div style="background: rgba(128, 128, 128, 0.05); border-left: 6px solid {border_color}; padding: 20px; border-radius: 8px; margin-bottom: 15px;">
            <h4>{status_icon} Question {idx + 1}</h4>
            <p><b>{q_data['question']}</b></p>
            <p><b>Your Answer:</b> {user_ans}</p>
            <p><b>Correct Answer:</b> {correct_ans}</p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.expander("📖 View Explanation"):
            steps = q_data.get("explanation_steps", [])
            if isinstance(steps, list):
                for step in steps:
                    st.markdown(f"**{step}**")
                    st.write("")
            else:
                st.markdown(steps)
                
            shortcut = q_data.get("shortcut_method", "")
            if shortcut:
                st.info(f"💡 **{shortcut}**")
    
    st.markdown("---")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric(label="Final Score", value=f"{score} / {total_q}")
    with col2:
        percentage = (score / total_q) * 100
        if percentage >= 80:
            st.success(f"Excellent work! You scored {percentage:.0f}%.")
        elif percentage >= 50:
            st.warning(f"Good effort. You scored {percentage:.0f}%. Review the explanations below.")
        else:
            st.error(f"You scored {percentage:.0f}%. Keep practicing!")
            
    st.markdown("---")
    if st.button("🔄 Take Another Test"):
        st.session_state.quiz_data = []
        st.session_state.current_q_idx = 0
        st.session_state.quiz_submitted = False
        st.session_state.user_answers = {}
        st.rerun()
