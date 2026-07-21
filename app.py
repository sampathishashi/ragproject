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
# 1. CONFIGURATION & SETUP (SECURE)
# ==========================================
# Ensure you set the GROQ_API_KEY in your Streamlit Cloud secrets or local environment
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("🚨 GROQ_API_KEY not found. Please set it as an environment variable before running.")

# Configuration variables
DEFAULT_COMPANY = "Google"
DEFAULT_TOPIC = "Data Structures and Quantitative Aptitude"
OUTPUT_FILE = "aptitude_questions.txt"
NUM_QUESTIONS = 25
MAX_WORKERS = 8  # Number of parallel threads (keep 4-8 for free Groq tier)

# ==========================================
# 2. DATA INGESTION & VECTOR STORE
# ==========================================
# Create the study material file
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

with open("study_material.txt", "w", encoding="utf-8") as f:
    f.write(sample_text.strip())

# Load and split the text
with open("study_material.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_text(raw_text)

# Initialize embeddings and FAISS vector store
print("Initializing Embeddings and Vector Store...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = FAISS.from_texts(chunks, embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 2})

# ==========================================
# 3. LLM & PROMPT ENGINEERING
# ==========================================
# Initialize Groq LLM
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.7)

# Helper function to format retrieved documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Define the prompt template
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

# Extract query for retriever
retrieval_query_extractor = RunnableLambda(lambda inputs: f"{inputs['topic']} {inputs['sub_topic']}")

# Assemble the RAG pipeline
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
# Retry mechanism to handle API rate limits
@tenacity.retry(
    wait=tenacity.wait_exponential(multiplier=1, min=4, max=60),
    stop=tenacity.stop_after_attempt(5),
    reraise=True,
)
def invoke_with_retry(inputs):
    return aptitude_chain.invoke(inputs)

# Sub-topic pool to ensure question variety
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

# Worker function for thread pool
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
# 5. MAIN EXECUTION FUNCTION
# ==========================================
def run_generation(company, topic, num_questions=25, max_workers=8):
    # Clear previous output file
    open(OUTPUT_FILE, "w", encoding="utf-8").close()

    print(f"\n🚀 Generating {num_questions} {company}-style '{topic}' questions in PARALLEL...")
    start_time = time.time()
    results = {}

    # Execute generation in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(generate_one, i, company, topic)
            for i in range(1, num_questions + 1)
        ]
        for future in as_completed(futures):
            idx, sub, result, error = future.result()
            if error:
                print(f"❌ Q{idx} ({sub}) failed: {error[:80]}")
                results[idx] = (sub, f"FAILED: {error}")
            else:
                print(f"✅ Q{idx}/{num_questions} done ({sub})")
                results[idx] = (sub, result)

    # Write all results in exact sequential order
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for i in range(1, num_questions + 1):
            if i in results:
                sub, text = results[i]
                f.write(f"--- Question {i}/{num_questions}  [Focus: {sub}] ---\n")
                f.write(text)
                f.write("\n" + "=" * 60 + "\n\n")
            else:
                f.write(f"--- Question {i}/{num_questions}: MISSING ---\n\n")

    elapsed = time.time() - start_time
    print(f"\n🎉 Done! {num_questions} questions generated in {elapsed:.1f}s")
    print(f"📁 Output saved to: {OUTPUT_FILE}")

# ==========================================
# 6. RUN THE CODE
# ==========================================
if __name__ == "__main__":
    run_generation(
        company=DEFAULT_COMPANY, 
        topic=DEFAULT_TOPIC, 
        num_questions=NUM_QUESTIONS, 
        max_workers=MAX_WORKERS
    )
