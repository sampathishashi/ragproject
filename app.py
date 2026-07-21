import streamlit as st
import random
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

.block-container{
    padding-top:1rem;
    padding-bottom:2rem;
}

.main-title{
    font-size:40px;
    font-weight:700;
    color:#2E8B57;
}

.sub-title{
    color:#808080;
    font-size:17px;
}

.card{
    background:#f8f9fa;
    padding:18px;
    border-radius:10px;
    border:1px solid #e3e3e3;
}

.stButton>button{
    width:100%;
    height:50px;
    font-size:18px;
    font-weight:bold;
    border-radius:8px;
}

.stDownloadButton>button{
    width:100%;
}

</style>
""", unsafe_allow_html=True)

# ==========================================================
# HEADER
# ==========================================================

st.markdown(
"""
<div class="main-title">
🎯 AI Mock Test Generator
</div>

<div class="sub-title">
Professional Interview Question Generator using
<b>RAG + FAISS + HuggingFace + Groq</b>
</div>
""",
unsafe_allow_html=True
)

st.divider()

# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.title("⚙ Configuration")

company = st.sidebar.selectbox(
    "Target Company",
    [
        "Google",
        "Amazon",
        "Microsoft",
        "Adobe",
        "Oracle",
        "Uber",
        "Goldman Sachs",
        "Flipkart",
        "Infosys",
        "TCS",
        "Accenture"
    ]
)

question_type = st.sidebar.selectbox(
    "Question Type",
    [
        "Aptitude",
        "Coding"
    ]
)

difficulty = st.sidebar.selectbox(
    "Difficulty",
    [
        "Easy",
        "Medium",
        "Hard"
    ]
)

num_questions = st.sidebar.slider(
    "Number of Questions",
    1,
    20,
    5
)

topic = st.text_input(
    "Enter Topic",
    placeholder="Arrays, Percentage, DBMS..."
)

generate = st.button("🚀 Generate Questions")

# ==========================================================
# SESSION STATE
# ==========================================================

if "history" not in st.session_state:
    st.session_state.history = []

if "generated_questions" not in st.session_state:
    st.session_state.generated_questions = set()

# ==========================================================
# BUILT-IN KNOWLEDGE BASE
# ==========================================================

study_material = """

DATA STRUCTURES

Arrays
Searching
Sorting
Binary Search
Sliding Window
Two Pointer
Prefix Sum
Kadane Algorithm

Strings
Palindrome
Anagram
KMP
Rabin Karp
Z Algorithm

Linked List
Stack
Queue
Heap
Trie

Trees
Binary Tree
BST
AVL
Segment Tree

Graphs
DFS
BFS
Topological Sort
Dijkstra
Bellman Ford
Union Find
Minimum Spanning Tree

Dynamic Programming
0/1 Knapsack
LCS
LIS
Coin Change
Matrix Chain Multiplication

DBMS
Normalization
Transactions
SQL
Joins
Indexing

Operating System
CPU Scheduling
Deadlock
Paging
Segmentation
Memory Management

Computer Networks
TCP
UDP
HTTP
HTTPS
DNS
Routing
OSI Model

Object Oriented Programming
Inheritance
Encapsulation
Polymorphism
Abstraction

QUANTITATIVE APTITUDE

Percentage
Profit and Loss
Simple Interest
Compound Interest
Average
Ratio and Proportion
Probability
Permutation
Combination
Number System
Time and Work
Time Speed Distance
Boats and Streams
Mixtures
Pipes and Cisterns

Logical Reasoning
Blood Relations
Directions
Coding Decoding
Seating Arrangement
Syllogism

Data Interpretation
Bar Graph
Pie Chart
Table
Line Graph

"""
# ==========================================================
# VECTOR DATABASE
# ==========================================================

@st.cache_resource
def load_vector_database():

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100
    )

    chunks = splitter.split_text(study_material)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    vector_db = FAISS.from_texts(
        texts=chunks,
        embedding=embeddings
    )

    return vector_db


vector_db = load_vector_database()

# ==========================================================
# RETRIEVER
# ==========================================================

retriever = vector_db.as_retriever(

    search_type="mmr",

    search_kwargs={

        "k": 6,

        "fetch_k": 20

    }

)

# ==========================================================
# LLM CONFIGURATION
# ==========================================================

llm = ChatGroq(

    api_key=st.secrets["GROQ_API_KEY"],

    model="llama-3.3-70b-versatile",

    temperature=0.2,

    max_tokens=4096

)

# ==========================================================
# CONTEXT RETRIEVAL
# ==========================================================

def retrieve_context(topic: str):

    docs = retriever.invoke(topic)

    if not docs:
        return ""

    return "\n\n".join(
        doc.page_content
        for doc in docs
    )

# ==========================================================
# CLEAN RESPONSE
# ==========================================================

def clean_response(text):

    if not text:
        return ""

    text = text.replace("```", "")

    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()

# ==========================================================
# RANDOM QUESTION STYLES
# ==========================================================

QUESTION_STYLES = [

    "Google Online Assessment",

    "Amazon Online Assessment",

    "Microsoft Coding Round",

    "Adobe Hiring Test",

    "Uber Technical Assessment",

    "Goldman Sachs Aptitude Test",

    "Campus Placement Test",

    "Competitive Programming",

    "Interview Screening Test"

]

# ==========================================================
# GENERATION TIMER
# ==========================================================

def generation_time():

    return datetime.now().strftime("%d-%m-%Y %H:%M:%S")

# ==========================================================
# APP STATUS
# ==========================================================

st.sidebar.markdown("---")

st.sidebar.success("✅ AI Engine Ready")

st.sidebar.write("LLM : Groq")

st.sidebar.write("Embeddings : MiniLM")

st.sidebar.write("Vector DB : FAISS")

st.sidebar.write("Retriever : MMR")

st.sidebar.write("Status : Ready")

st.sidebar.markdown("---")
