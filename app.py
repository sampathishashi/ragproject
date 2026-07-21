import os
import time
import json
import re
import streamlit as st
import tenacity
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS

# ==========================================
# 1. SECURE API KEY SETUP FOR STREAMLIT
# ==========================================
try:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except KeyError:
    st.error("🚨 GROQ_API_KEY not found! Please go to your app settings and add it to Secrets.")
    st.stop()

# ==========================================
# 2. CACHED DATA & VECTOR STORE
# ==========================================
@st.cache_resource
def get_vector_store():
    sample_text = """
    Data Structures & Algorithms:
    Arrays are collections of elements stored at contiguous memory locations. Common operations include traversal, insertion, deletion, searching (linear and binary search), and sorting.
    Two-pointer approaches, sliding windows, prefix sums, and sliding matrices are powerful optimizations for array challenges.
    Dynamic Programming (DP) is applicable to sequential array problems exhibiting overlapping subproblems and optimal substructures, such as the 0/1 Knapsack problem or Longest Increasing Subsequence.
    Quantitative Aptitude:
    Permutations and Combinations. A permutation is an arrangement of outcomes where order matters. A combination is a selection of outcomes where order does not matter.
    Formula for Permutation: nPr = n! / (n - r)!
    Formula for Combination: nCr = n! / (r! * (n - r)!)
    Probability: P(Event) = Favorable Outcomes / Total Outcomes.
    Time, Speed & Distance: Distance = Speed × Time.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_text(sample_text)
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(chunks, embeddings)
    return vector_store.as_retriever(search_kwargs={"k": 2})

retriever = get_vector_store()

# ==========================================
# 3. LLM & PROMPT ENGINEERING (STRICT JSON OUTPUT)
# ==========================================
@st.cache_resource
def get_llm():
    return ChatGroq(model="llama-3.1-8b-instant", temperature=0.7)

llm = get_llm()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

aptitude_prompt = ChatPromptTemplate.from_template("""
You are an expert technical interviewer for {company_name}.
Create exactly 1 highly unique multiple-choice aptitude question.

Target Main Topic: {topic}
Sub-Focus Area: {sub_topic}
Context: {context}

CRITICAL UNIQUENESS RULE: Use the seed to make this question DIFFERENT. Seed: {seed}

Output STRICTLY a valid JSON object in this exact format. Do NOT output any markdown or text before or after the JSON:
{{
  "question": "The question text here?",
  "options": {{
    "A": "Option A text",
    "B": "Option B text",
    "C": "Option C text",
    "D": "Option D text"
  }},
  "correct_answer": "A",
  "explanation": "Step-by-step logical or mathematical explanation."
}}
""")

retrieval_query_extractor = RunnableLambda(lambda inputs: f"{inputs['topic']} {inputs['sub_topic']}")

aptitude_chain = (
    {
        "context": retrieval_query_extractor | retriever | format_docs,
        "company_name": lambda x: x.get("company_name", "a top tech firm"),
        "topic": lambda x: x["topic"],
        "sub_topic": lambda x: x["sub_topic"],
        "seed": lambda x: x["seed"],
    }
    | aptitude_prompt | llm | StrOutputParser()
)

# ==========================================
# 4. PARALLEL EXECUTION WORKERS
# ==========================================
@tenacity.retry(
    wait=tenacity.wait_exponential(multiplier=1, min=4, max=60),
    stop=tenacity.stop_after_attempt(5),
    reraise=True,
)
def invoke_with_retry(inputs):
    return aptitude_chain.invoke(inputs)

SUB_TOPICS = [
    "basic concept", "edge case", "real-world application", "advanced variation",
    "computation heavy", "trick / common pitfall", "time-complexity analysis",
    "data interpretation", "multi-step reasoning", "formula application",
    "reverse calculation", "boundary condition", "optimization angle",
    "comparative analysis", "theoretical foundation", "debugging scenario",
    "estimation problem", "pattern recognition", "logical deduction",
    "verbal reasoning twist", "numeric series", "geometric interpretation",
    "probability flavor", "combinatorial count", "speed-distance variant",
]

def generate_one(idx, company, topic):
    sub = SUB_TOPICS[idx % len(SUB_TOPICS)]
    seed = f"Q{idx}-{int(time.time())}-{idx*7919}"
    inputs = {
        "company_name": company,
        "topic": topic,
        "sub_topic": sub,
        "seed": seed,
    }
    try:
        result = invoke_with_retry(inputs)
        # Parse JSON safely
        json_match = re.search(r"\{.*\}", result, re.DOTALL)
        if json_match:
            q_data = json.loads(json_match.group(0))
            # Validate required fields
            if all(k in q_data for k in ["question", "options", "correct_answer", "explanation"]):
                return idx, sub, q_data, None
        return idx, sub, None, "JSON Parse Error"
    except Exception as e:
        return idx, sub, None, str(e)

# ==========================================
# 5. SESSION STATE INITIALIZATION
# ==========================================
if 'exam_questions' not in st.session_state:
    st.session_state.exam_questions = []
    st.session_state.exam_submitted = False
    st.session_state.score = 0

# ==========================================
# 6. STREAMLIT UI & MAIN EXECUTION
# ==========================================
st.title("📝 Aptitude Test Generator & Evaluator")
st.markdown("Generate a test, answer the questions, and submit to see your marks and explanations.")

col1, col2 = st.columns(2)
with col1:
    company = st.text_input("Company Name", "Google")
with col2:
    topic = st.text_input("Topic", "Data Structures and Aptitude")

num_questions = st.slider("Number of Questions", min_value=5, max_value=25, value=5, step=5)
max_workers = st.slider("Generation Speed (Parallel Workers)", min_value=2, max_value=8, value=4)

# --- STEP 1: GENERATE EXAM ---
if st.button("🚀 Generate Exam"):
    st.info(f"Generating {num_questions} questions for {company}...")
    
    start_time = time.time()
    results = {}
    progress_bar = st.progress(0)
    status_text = st.empty()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(generate_one, i, company, topic)
            for i in range(1, num_questions + 1)
        ]
        
        completed = 0
        for future in as_completed(futures):
            idx, sub, result, error = future.result()
            if error:
                st.error(f"Failed to generate Q{idx}: {error}")
            else:
                results[idx] = result
            
            completed += 1
            progress_bar.progress(completed / num_questions)
            status_text.text(f"Generated {completed}/{num_questions}...")

    elapsed = time.time() - start_time
    
    # Sort questions by index
    final_questions = [results[i] for i in sorted(results.keys()) if results[i] is not None]
    
    if final_questions:
        st.session_state.exam_questions = final_questions
        st.session_state.exam_submitted = False
        st.session_state.score = 0
        st.success(f"🎉 Exam generated in {elapsed:.1f}s! Please answer the questions below.")
        st.rerun()
    else:
        st.error("Failed to generate questions. Please try again.")

# --- STEP 2: TAKE THE EXAM (DISPLAY QUESTIONS & OPTIONS ONLY) ---
if st.session_state.exam_questions and not st.session_state.exam_submitted:
    st.markdown("---")
    st.subheader("📋 Assignment Test")
    st.markdown("*Select the best answer for each question.*")
    
    with st.form("exam_form"):
        for i, q_data in enumerate(st.session_state.exam_questions):
            st.markdown(f"**Q{i+1}: {q_data['question']}**")
            options = [f"{k}: {v}" for k, v in q_data['options'].items()]
            st.radio("Your Answer:", options, key=f"q_{i}", index=None)
            st.markdown("---")
        
        submitted = st.form_submit_button("✅ Submit Exam")
        if submitted:
            st.session_state.exam_submitted = True
            st.rerun()

# --- STEP 3: SHOW RESULTS, MARKS, EXPLANATIONS ---
if st.session_state.exam_submitted:
    st.markdown("---")
    st.subheader("📊 Test Results & Explanations")
    
    score = 0
    total = len(st.session_state.exam_questions)
    
    for i, q_data in enumerate(st.session_state.exam_questions):
        user_choice_str = st.session_state.get(f"q_{i}", "No Answer")
        user_ans = user_choice_str.split(":")[0] if user_choice_str != "No Answer" else "No Answer"
        correct_ans = q_data['correct_answer'].strip().upper()
        
        st.markdown(f"**Q{i+1}: {q_data['question']}**")
        
        if user_ans == correct_ans:
            st.success(f"✅ Your Answer: {user_ans} (Correct!)")
            score += 1
        else:
            st.error(f"❌ Your Answer: {user_ans} (Incorrect)")
            st.success(f"✔️ Correct Answer: {correct_ans}")
            
        st.info(f"**Explanation:** {q_data['explanation']}")
        st.markdown("---")
    
    st.session_state.score = score
    
    # Final Score Display
    st.metric("Your Final Score", f"{score} / {total}")
    percentage = (score / total) * 100
    if percentage >= 80:
        st.balloons()
        st.success(f"Excellent! You scored {percentage:.1f}%")
    elif percentage >= 50:
        st.warning(f"Good effort! You scored {percentage:.1f}%")
    else:
        st.error(f"Needs improvement. You scored {percentage:.1f}%")

    if st.button("🔄 Start New Exam"):
        st.session_state.exam_questions = []
        st.session_state.exam_submitted = False
        st.session_state.score = 0
        st.rerun()
