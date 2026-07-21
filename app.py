import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# ----------------------------
# PAGE CONFIG
# ----------------------------

st.set_page_config(
    page_title="AI Mock Test Generator",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 AI Mock Test Generator")

st.markdown(
"""
Generate Company Specific

✅ Aptitude Questions

✅ Coding Questions

using

LangChain + FAISS + Groq
"""
)

# ----------------------------
# SIDEBAR
# ----------------------------

st.sidebar.header("Settings")

company = st.sidebar.selectbox(
    "Company",
    [
        "Google",
        "Amazon",
        "Microsoft",
        "Adobe",
        "Oracle",
        "Uber",
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
    10,
    3
)

topic = st.text_input(
    "Enter Topic",
    placeholder="Arrays"
)

generate = st.button("Generate Questions")

# ----------------------------
# BUILT-IN STUDY MATERIAL
# ----------------------------

study_material = """

DATA STRUCTURES

Arrays
Searching
Sorting
Prefix Sum
Sliding Window
Two Pointer
Kadane Algorithm

Strings
Palindrome
KMP
Z Algorithm
Hashing
Pattern Matching

Linked List
Stack
Queue
Deque
Priority Queue

Trees
Binary Tree
BST
AVL
Heap
Trie

Graphs
DFS
BFS
Topological Sort
Shortest Path
Minimum Spanning Tree
Union Find

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
Indexing
Joins
ER Diagram

Operating System

CPU Scheduling
Deadlock
Memory Management
Paging
Segmentation

Computer Networks

OSI Model
TCP
UDP
IP
DNS
HTTP
HTTPS
Routing

OOPS

Inheritance
Polymorphism
Abstraction
Encapsulation
Constructor
Virtual Function

APTITUDE

Percentage

Profit and Loss

Simple Interest

Compound Interest

Average

Ratio

Probability

Permutation

Combination

Number System

Time and Work

Time Speed Distance

Pipes and Cisterns

Blood Relations

Coding Decoding

Logical Reasoning

Verbal Ability

Grammar

Reading Comprehension

Data Interpretation

Bar Graph

Pie Chart

Line Graph

"""

# ----------------------------
# CREATE VECTOR DATABASE
# ----------------------------

@st.cache_resource
def load_vector_db():

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_text(study_material)

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    db = FAISS.from_texts(
        chunks,
        embeddings
    )

    return db

db = load_vector_db()

retriever = db.as_retriever(
    search_kwargs={"k":3}
)

# ----------------------------
# GROQ MODEL
# ----------------------------

llm = ChatGroq(
    api_key=st.secrets["GROQ_API_KEY"],
    model="llama-3.1-8b-instant",
    temperature=0.7
)

# ==========================================================
# PROMPT TEMPLATES
# ==========================================================

aptitude_prompt = ChatPromptTemplate.from_template("""
You are an expert aptitude interviewer for {company}.

Use ONLY the given context.

Context:
{context}

Generate ONE {difficulty} level aptitude question.

Topic:
{topic}

Requirements:

1. Question
2. Four Options (A,B,C,D)
3. Correct Answer
4. Detailed Explanation
5. Don't mention the context.
""")

coding_prompt = ChatPromptTemplate.from_template("""
You are an expert coding interviewer for {company}.

Use ONLY the given context.

Context:
{context}

Generate ONE {difficulty} coding interview question.

Topic:
{topic}

Requirements:

1. Problem Statement
2. Input Format
3. Output Format
4. Constraints
5. Sample Input
6. Sample Output
7. Python Function
8. Time Complexity
9. Space Complexity
10. Hint
11. Explanation
""")

# ==========================================================
# RETRIEVE CONTEXT
# ==========================================================

def retrieve_context(query):

    docs = retriever.invoke(query)

    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    return context


# ==========================================================
# GENERATE QUESTION
# ==========================================================

def generate_question(
    company,
    topic,
    difficulty,
    question_type
):

    context = retrieve_context(topic)

    if question_type == "Aptitude":

        prompt = aptitude_prompt.format(
            company=company,
            topic=topic,
            difficulty=difficulty,
            context=context
        )

    else:

        prompt = coding_prompt.format(
            company=company,
            topic=topic,
            difficulty=difficulty,
            context=context
        )

    response = llm.invoke(prompt)

    return response.content
    # ==========================================================
# GENERATE QUESTIONS
# ==========================================================

if generate:

    if topic.strip() == "":
        st.error("Please enter a topic.")
        st.stop()

    st.divider()

    st.subheader("📄 Generated Questions")

    progress = st.progress(0)

    output_text = ""

    for i in range(1, num_questions + 1):

        progress.progress(i / num_questions)

        with st.spinner(f"Generating Question {i}..."):

            result = generate_question(
                company=company,
                topic=topic,
                difficulty=difficulty,
                question_type=question_type
            )

        with st.expander(f"Question {i}", expanded=True):

            st.markdown(result)

        output_text += (
            f"\n\n==============================\n"
            f"Question {i}\n"
            f"==============================\n\n"
        )

        output_text += result
        output_text += "\n"

    st.success("✅ All Questions Generated Successfully!")

    st.download_button(
        label="📥 Download Questions",
        data=output_text,
        file_name=f"{company}_{topic}_{question_type}.txt",
        mime="text/plain"
    )

# ==========================================================
# SESSION STATE
# ==========================================================

if "history" not in st.session_state:
    st.session_state.history = []

# ==========================================================
# SAVE HISTORY
# ==========================================================

if generate and topic.strip() != "":

    st.session_state.history.append(
        {
            "Company": company,
            "Topic": topic,
            "Type": question_type,
            "Difficulty": difficulty,
            "Questions": num_questions
        }
    )

# ==========================================================
# SIDEBAR STATISTICS
# ==========================================================

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Statistics")

st.sidebar.metric(
    "Generated Sessions",
    len(st.session_state.history)
)

st.sidebar.metric(
    "Current Questions",
    num_questions
)

# ==========================================================
# GENERATION HISTORY
# ==========================================================

if len(st.session_state.history) > 0:

    st.divider()

    st.subheader("📝 Generation History")

    for i, item in enumerate(reversed(st.session_state.history), 1):

        with st.expander(f"History {i}"):

            st.write(f"**Company:** {item['Company']}")
            st.write(f"**Topic:** {item['Topic']}")
            st.write(f"**Question Type:** {item['Type']}")
            st.write(f"**Difficulty:** {item['Difficulty']}")
            st.write(f"**Questions:** {item['Questions']}")

# ==========================================================
# CLEAR HISTORY
# ==========================================================

if st.sidebar.button("🗑 Clear History"):

    st.session_state.history = []

    st.rerun()

# ==========================================================
# APP INFO
# ==========================================================

st.sidebar.markdown("---")

st.sidebar.info(
"""
### AI Mock Test Generator

Features

✅ RAG

✅ FAISS

✅ HuggingFace Embeddings

✅ Groq LLM

✅ Aptitude Generator

✅ Coding Generator

✅ Company-wise Questions

✅ Download Questions
"""
)

# ==========================================================
# FOOTER
# ==========================================================

st.divider()

st.caption(
"🚀 Built using Streamlit • LangChain • FAISS • HuggingFace • Groq"
)
import random
import time

# ==========================================================
# RANDOM QUESTION STYLE
# ==========================================================

QUESTION_STYLES = [
    "real campus placement question",
    "company online assessment question",
    "interview round question",
    "coding assessment question",
    "advanced interview question",
    "fresh unique question",
    "competitive programming style",
    "logical reasoning style"
]

# ==========================================================
# RANDOMIZE PROMPT
# ==========================================================

def generate_question(
    company,
    topic,
    difficulty,
    question_type
):

    context = retrieve_context(topic)

    style = random.choice(QUESTION_STYLES)

    if question_type == "Aptitude":

        prompt = f"""
You are an expert aptitude interviewer.

Company:
{company}

Question Style:
{style}

Difficulty:
{difficulty}

Topic:
{topic}

Context:
{context}

Generate ONE completely NEW aptitude question.

Requirements:

• Never repeat previous questions.
• Four options (A,B,C,D)
• Correct Answer
• Detailed Explanation
"""

    else:

        prompt = f"""
You are an expert coding interviewer.

Company:
{company}

Question Style:
{style}

Difficulty:
{difficulty}

Topic:
{topic}

Context:
{context}

Generate ONE unique coding interview problem.

Include

Problem Statement

Constraints

Input Format

Output Format

Sample Input

Sample Output

Python Function

Hint

Time Complexity

Space Complexity
"""

    try:

        response = llm.invoke(prompt)

        return response.content

    except Exception as e:

        return f"❌ Error\n\n{e}"


# ==========================================================
# LOADING ANIMATION
# ==========================================================

def loading():

    bar = st.progress(0)

    for i in range(100):

        time.sleep(0.01)

        bar.progress(i + 1)

    bar.empty()


# ==========================================================
# DISPLAY HEADER
# ==========================================================

def result_header():

    st.success("Questions Generated Successfully!")

    st.balloons()
    # ==========================================================
# PROFESSIONAL CSS
# ==========================================================

st.markdown("""
<style>

/* Main background */

.stApp{
    background-color:#0E1117;
}

/* Title */

h1{
    color:#00FFD1;
    text-align:center;
}

/* Sub headers */

h2,h3{
    color:white;
}

/* Sidebar */

section[data-testid="stSidebar"]{
    background:#161B22;
}

/* Buttons */

.stButton>button{
    width:100%;
    height:50px;
    background:#00FFD1;
    color:black;
    font-size:18px;
    font-weight:bold;
    border-radius:10px;
}

.stButton>button:hover{
    background:#00C9A7;
    color:white;
}

/* Download button */

.stDownloadButton>button{
    width:100%;
    height:45px;
    border-radius:8px;
}

/* Expander */

.streamlit-expanderHeader{
    font-size:18px;
    font-weight:bold;
}

/* Metric Cards */

[data-testid="metric-container"]{
    background:#161B22;
    border-radius:12px;
    padding:10px;
}

/* Text Input */

input{
    border-radius:8px !important;
}

</style>
""", unsafe_allow_html=True)

# ==========================================================
# HOME INFORMATION
# ==========================================================

st.markdown("---")

st.markdown("""
## 🚀 Features

- 🤖 Groq LLM
- 📚 LangChain
- 🧠 HuggingFace Embeddings
- ⚡ FAISS Vector Database
- 🎯 Company Specific Questions
- 💻 Coding Problems
- 📝 Aptitude Questions
- 📥 Download Generated Questions
- 🌙 Professional Dark Theme
""")

# ==========================================================
# ABOUT
# ==========================================================

with st.expander("ℹ About Project"):

    st.write("""
This project is an AI Powered Mock Test Generator.

Technology Stack

• Streamlit

• LangChain

• HuggingFace Embeddings

• FAISS

• Groq

• Retrieval Augmented Generation (RAG)

Supported Interview Types

• Google

• Amazon

• Microsoft

• Adobe

• Oracle

• TCS

• Infosys

• Accenture

• Uber

• Flipkart

This project generates interview questions using Retrieval Augmented Generation.
""")

# ==========================================================
# FOOTER
# ==========================================================

st.markdown("---")

st.markdown(
"""
<center>

### 🎯 AI Mock Test Generator

Built using ❤️

Streamlit • LangChain • FAISS • HuggingFace • Groq

© 2026 All Rights Reserved

</center>
""",
unsafe_allow_html=True
)
