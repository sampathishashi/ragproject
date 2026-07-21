import sys
from unittest.mock import MagicMock

# ==========================================
# 0. FIX FOR STREAMLIT CLOUD LOG SPAM
# Streamlit's watcher scans 'transformers' image models which require 'torchvision'.
# By mocking it, we prevent hundreds of ModuleNotFoundError warnings in the logs.
# ==========================================
if 'torchvision' not in sys.modules:
    sys.modules['torchvision'] = MagicMock()

import os
import time
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
# Using st.cache_resource so it only loads once when the app starts
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
# 3. LLM & PROMPT ENGINEERING
# ==========================================
@st.cache_resource
def get_llm():
    return ChatGroq(model="llama-3.1-8b-instant", temperature=0.7)

llm = get_llm()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

aptitude_prompt = ChatPromptTemplate.from_template("""
You are an expert technical interviewer for {company_name}.
Use the provided context to create exactly 1 highly unique multiple-choice aptitude question
following {company_name}'s typical recruitment style.

Target Main Topic: {topic}
Specific Sub-Focus Area: {sub_topic}
Question Number: {question_num}

Context:
{context}

CRITICAL UNIQUENESS RULE:
- Use the seed below to make this question DIFFERENT from every other question.
- Do NOT reuse the same names, numbers, or scenarios.
- Seed (random seed for this question): {seed}

Requirements:
1. Formulate a challenging scenario tied to the sub-focus area.
2. Provide 4 distinct options (A, B, C, D).
3. Clearly state the Correct Answer.
4. Provide a step-by-step mathematical or logical Explanation.
""")

retrieval_query_extractor = RunnableLambda(lambda inputs: f"{inputs['topic']} {inputs['sub_topic']}")

aptitude_chain = (
    {
        "context": retrieval_query_extractor | retriever | format_docs,
        "company_name": lambda x: x.get("company_name", "a top tech firm"),
        "topic": lambda x: x["topic"],
        "sub_topic": lambda x: x["sub_topic"],
        "question_num": lambda x: x["question_num"],
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
        "question_num": str(idx),
        "seed": seed,
    }
    try:
        result = invoke_with_retry(inputs)
        return idx, sub, result, None
    except Exception as e:
        return idx, sub, None, str(e)

# ==========================================
# 5. STREAMLIT UI & MAIN EXECUTION
# ==========================================
st.title("📝 Aptitude Question Generator (RAG)")
st.markdown("Generate bulk multiple-choice aptitude questions using Groq & LangChain.")

col1, col2 = st.columns(2)
with col1:
    company = st.text_input("Company Name", "Google")
with col2:
    topic = st.text_input("Topic", "Data Structures and Aptitude")

num_questions = st.slider("Number of Questions", min_value=5, max_value=25, value=10, step=5)
max_workers = st.slider("Parallel Workers (Speed)", min_value=2, max_value=8, value=4)

if st.button("🚀 Generate Questions"):
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
                results[idx] = (sub, f"❌ FAILED: {error}")
            else:
                results[idx] = (sub, result)
            
            completed += 1
            progress_bar.progress(completed / num_questions)
            status_text.text(f"Completed {completed}/{num_questions}...")

    elapsed = time.time() - start_time
    st.success(f"🎉 Done! Generated in {elapsed:.1f}s")

    # Display results in order
    st.subheader("Generated Questions")
    full_text = ""
    for i in range(1, num_questions + 1):
        if i in results:
            sub, text = results[i]
            st.markdown(f"**Question {i}/{num_questions} [Focus: {sub}]**")
            st.markdown(text)
            st.markdown("---")
            full_text += f"--- Question {i}/{num_questions}  [Focus: {sub}] ---\n{text}\n\n{'='*60}\n\n"

    # Provide download button
    st.download_button(
        label="📥 Download Questions as TXT",
        data=full_text,
        file_name=f"{company}_aptitude_questions.txt",
        mime="text/plain"
    )
