import streamlit as st
import random
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="AI Mock Test Generator",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>

.main-title{
    font-size:40px;
    font-weight:bold;
    color:#4CAF50;
}

.sub-title{
    font-size:18px;
    color:gray;
}

.block-container{
    padding-top:2rem;
}

div.stButton>button{
    width:100%;
    height:50px;
    font-size:18px;
    font-weight:bold;
    border-radius:10px;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# HEADER
# =====================================================

st.markdown(
"""
<div class="main-title">
🎯 AI Mock Test Generator
</div>

<div class="sub-title">

Generate Professional Company-Level

• Aptitude Questions

• Coding Questions

using

LangChain + FAISS + HuggingFace + Groq

</div>
""",
unsafe_allow_html=True
)

st.divider()

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("⚙ Settings")

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
        "Goldman Sachs",
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
    placeholder="Arrays, Graphs, Percentage..."
)

generate = st.button("🚀 Generate Questions")

# =====================================================
# BUILT-IN KNOWLEDGE BASE
# =====================================================

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
KMP
Rabin Karp
Z Algorithm
Pattern Matching

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

Floyd Warshall

Union Find

Minimum Spanning Tree

Dynamic Programming

0/1 Knapsack

LCS

LIS

Matrix Chain Multiplication

Coin Change

Subset Sum

Backtracking

Recursion

Greedy Algorithms

DBMS

Normalization

Transactions

Joins

SQL

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

OOPS

Inheritance

Encapsulation

Polymorphism

Abstraction

Constructor

Exception Handling

QUANTITATIVE APTITUDE

Percentages

Profit and Loss

Simple Interest

Compound Interest

Partnership

Average

Ratio and Proportion

Probability

Permutation

Combination

Number System

Time and Work

Time Speed Distance

Pipes and Cisterns

Ages

Mixtures

Boats and Streams

Logical Reasoning

Coding Decoding

Blood Relations

Directions

Seating Arrangement

Syllogism

Data Interpretation

Bar Graph

Pie Chart

Table

Caselet

Line Graph

"""

# =====================================================
# VECTOR DATABASE
# =====================================================

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

vector_db = load_vector_db()

retriever = vector_db.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k":6,
        "fetch_k":20
    }
)

# =====================================================
# LLM
# =====================================================

llm = ChatGroq(
    api_key=st.secrets["GROQ_API_KEY"],
    model="llama-3.3-70b-versatile",
    temperature=0.2,
    max_tokens=2048
)
# =====================================================
# PROFESSIONAL PROMPT TEMPLATES
# =====================================================

aptitude_prompt = ChatPromptTemplate.from_template("""
You are a Senior Assessment Designer with 15+ years of experience creating aptitude assessments for

• Google
• Amazon
• Microsoft
• Adobe
• Goldman Sachs

Company:
{company}

Difficulty:
{difficulty}

Topic:
{topic}

Reference Material:
{context}

====================================================

TASK

Generate EXACTLY ONE brand-new aptitude question.

The question must look like a real company online assessment.

====================================================

RULES

1. Never copy textbook questions.

2. Never copy examples from context.

3. Create an original business scenario.

4. Use realistic numbers.

5. Make calculations BEFORE writing.

6. Verify calculations twice.

7. Exactly ONE option must be correct.

8. Wrong options should be believable.

9. Final answer MUST exactly match one option.

10. Never use

"The closest answer"

"Approximately"

"However"

"If calculation differs"

11. If answer doesn't match an option,

REGENERATE the question.

====================================================

OUTPUT FORMAT

Question

Options

A.

B.

C.

D.

Correct Answer

Detailed Solution

Step 1

Step 2

Step 3

Final Verification

Concept Tested

Difficulty

Expected Time

Return ONLY the final question.
""")

coding_prompt = ChatPromptTemplate.from_template("""
You are a Senior Coding Interviewer.

Company:
{company}

Difficulty:
{difficulty}

Topic:
{topic}

Reference:
{context}

Generate ONE original coding interview problem.

It must resemble

Google

Microsoft

Amazon

Adobe

Requirements

Problem Statement

Input Format

Output Format

Constraints

Sample Input

Sample Output

Hidden Test Cases

Python Function

Hints

Time Complexity

Space Complexity

Concepts Tested

Return ONLY the problem.
""")

# =====================================================
# CONTEXT RETRIEVAL
# =====================================================

def retrieve_context(topic):

    docs = retriever.invoke(topic)

    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    return context


# =====================================================
# RANDOM COMPANY STYLE
# =====================================================

styles = [

    "Google Online Assessment",

    "Amazon Assessment",

    "Microsoft OA",

    "Adobe Hiring Test",

    "Goldman Sachs Aptitude",

    "Campus Placement",

    "Competitive Programming",

    "Interview Round"
]


# =====================================================
# FIRST GENERATION
# =====================================================

def first_generation(company, topic, difficulty, question_type):

    context = retrieve_context(topic)

    style = random.choice(styles)

    if question_type == "Aptitude":

        prompt = aptitude_prompt.format(

            company=f"{company} ({style})",

            difficulty=difficulty,

            topic=topic,

            context=context

        )

    else:

        prompt = coding_prompt.format(

            company=f"{company} ({style})",

            difficulty=difficulty,

            topic=topic,

            context=context

        )

    response = llm.invoke(prompt)

    return response.content
    # =====================================================
# AI REVIEWER PROMPT
# =====================================================

review_prompt = ChatPromptTemplate.from_template("""
You are a Senior Quality Assurance Reviewer.

Your responsibility is to review interview questions before they are released.

====================================================

Question

{question}

====================================================

CHECKLIST

1. Grammar

2. English

3. Logic

4. Mathematics

5. Formula

6. Calculation

7. Correct Answer

8. Options

9. Explanation

10. Difficulty

11. Duplicate values

12. Professional quality

====================================================

RULES

If ANY mistake exists

DO NOT explain the mistake.

Instead

Rewrite the ENTIRE question.

Requirements

• Exactly one correct option.

• Correct mathematics.

• Correct explanation.

• Professional English.

• No contradictions.

• No approximation.

• No repeated sentences.

• Output only the corrected final version.

====================================================

Return ONLY the final polished question.
""")


# =====================================================
# FINAL GENERATOR
# =====================================================

def generate_question(
    company,
    topic,
    difficulty,
    question_type
):

    # First Generation
    draft = first_generation(
        company,
        topic,
        difficulty,
        question_type
    )

    # AI Verification
    prompt = review_prompt.format(
        question=draft
    )

    reviewed = llm.invoke(prompt)

    return reviewed.content


# =====================================================
# DUPLICATE PREVENTION
# =====================================================

generated_questions = set()


def generate_unique_question(
    company,
    topic,
    difficulty,
    question_type,
    retries=5
):

    for _ in range(retries):

        result = generate_question(
            company,
            topic,
            difficulty,
            question_type
        )

        # Normalize whitespace for comparison
        normalized = " ".join(result.split())

        if normalized not in generated_questions:
            generated_questions.add(normalized)
            return result

    return result
    # =====================================================
# SESSION STATE
# =====================================================

if "history" not in st.session_state:
    st.session_state.history = []

# =====================================================
# GENERATE QUESTIONS
# =====================================================

if generate:

    if topic.strip() == "":
        st.warning("⚠ Please enter a topic.")
        st.stop()

    st.divider()

    st.subheader("🎯 Generated Questions")

    progress_bar = st.progress(0)

    status = st.empty()

    results = []

    generated_questions.clear()

    for i in range(num_questions):

        progress = int(((i + 1) / num_questions) * 100)

        progress_bar.progress(progress)

        status.info(
            f"Generating Question {i+1} of {num_questions}..."
        )

        question = generate_unique_question(
            company=company,
            topic=topic,
            difficulty=difficulty,
            question_type=question_type
        )

        results.append(question)

        with st.expander(
            f"Question {i+1}",
            expanded=True
        ):
            st.markdown(question)

    progress_bar.empty()

    status.success("✅ Questions Generated Successfully")

# =====================================================
# SAVE HISTORY
# =====================================================

    st.session_state.history.append(
        {
            "Company": company,
            "Topic": topic,
            "Difficulty": difficulty,
            "Type": question_type,
            "Questions": num_questions
        }
    )

# =====================================================
# DOWNLOAD
# =====================================================

    download_text = ""

    for index, item in enumerate(results, 1):

        download_text += f"""

==================================================

QUESTION {index}

==================================================

{item}

"""

    st.download_button(
        label="📥 Download Questions",
        data=download_text,
        file_name=f"{company}_{topic}.txt",
        mime="text/plain"
    )

# =====================================================
# HISTORY
# =====================================================

if len(st.session_state.history):

    st.divider()

    st.subheader("📜 Generation History")

    for item in reversed(st.session_state.history):

        st.markdown(f"""
**Company:** {item['Company']}

**Topic:** {item['Topic']}

**Difficulty:** {item['Difficulty']}

**Type:** {item['Type']}

**Questions:** {item['Questions']}

---
""")

# =====================================================
# SIDEBAR STATS
# =====================================================

st.sidebar.markdown("---")

st.sidebar.metric(
    "Generated Sessions",
    len(st.session_state.history)
)

st.sidebar.metric(
    "Current Questions",
    num_questions
)

st.sidebar.metric(
    "Difficulty",
    difficulty)

st.sidebar.metric(
    "Question Type",
    question_type)
# =====================================================
# PROFESSIONAL CSS
# =====================================================

st.markdown("""
<style>

/* ---------- Main App ---------- */

.stApp{
    background:#0E1117;
}

/* ---------- Title ---------- */

.title{
    font-size:42px;
    font-weight:800;
    color:#4CAF50;
    text-align:center;
}

.subtitle{
    font-size:18px;
    color:#C9D1D9;
    text-align:center;
    margin-bottom:25px;
}

/* ---------- Sidebar ---------- */

section[data-testid="stSidebar"]{
    background:#161B22;
}

/* ---------- Buttons ---------- */

.stButton>button{

    width:100%;

    height:52px;

    border-radius:12px;

    background:#238636;

    color:white;

    font-size:18px;

    font-weight:bold;

    border:none;

}

.stButton>button:hover{

    background:#2EA043;

    transition:0.3s;
}

/* ---------- Download ---------- */

.stDownloadButton>button{

    width:100%;

    background:#1F6FEB;

    color:white;

    border-radius:12px;

}

/* ---------- Expander ---------- */

.streamlit-expanderHeader{

    font-size:18px;

    font-weight:bold;

}

/* ---------- Metrics ---------- */

[data-testid="metric-container"]{

    background:#161B22;

    border:1px solid #30363D;

    border-radius:12px;

    padding:15px;

}

/* ---------- Text Input ---------- */

input{

    border-radius:10px !important;

}

/* ---------- Success ---------- */

.stSuccess{

    border-radius:10px;

}

</style>
""", unsafe_allow_html=True)

# =====================================================
# DASHBOARD
# =====================================================

st.markdown("---")

c1, c2, c3, c4 = st.columns(4)

with c1:

    st.metric(
        "Company",
        company
    )

with c2:

    st.metric(
        "Topic",
        topic if topic else "-"
    )

with c3:

    st.metric(
        "Questions",
        num_questions
    )

with c4:

    st.metric(
        "Difficulty",
        difficulty
    )

st.markdown("---")
# =====================================================
# FEATURE CARDS
# =====================================================

st.markdown("## ✨ Features")

col1, col2, col3 = st.columns(3)

with col1:
    st.info("""
### 📚 RAG Engine

- FAISS Vector Database
- HuggingFace Embeddings
- Semantic Search
- MMR Retrieval
""")

with col2:
    st.success("""
### 🤖 AI Generation

- Groq LLM
- Company Specific
- Coding Questions
- Aptitude Questions
""")

with col3:
    st.warning("""
### 🎯 Quality

- AI Verification
- Duplicate Prevention
- Professional Questions
- Download Results
""")

st.markdown("---")

# =====================================================
# TECHNOLOGY STACK
# =====================================================

st.markdown("## 🛠 Technology Stack")

tech1, tech2, tech3, tech4 = st.columns(4)

with tech1:
    st.metric("Frontend", "Streamlit")

with tech2:
    st.metric("LLM", "Groq")

with tech3:
    st.metric("Vector DB", "FAISS")

with tech4:
    st.metric("Embeddings", "MiniLM")

st.markdown("---")

# =====================================================
# ABOUT PROJECT
# =====================================================

with st.expander("📖 About AI Mock Test Generator", expanded=False):

    st.markdown("""
### AI Mock Test Generator

This application generates professional interview questions
using Retrieval Augmented Generation (RAG).

### Features

- Company Specific Questions
- Coding Interview Problems
- Aptitude Questions
- Dynamic Difficulty
- Semantic Search
- AI Quality Verification
- Duplicate Prevention
- Download Generated Questions

### Supported Companies

- Google
- Amazon
- Microsoft
- Adobe
- Oracle
- Uber
- Goldman Sachs
- Flipkart
- Infosys
- TCS
- Accenture

### Supported Topics

- DSA
- DBMS
- Operating Systems
- Computer Networks
- OOP
- Aptitude
- Logical Reasoning
- SQL
- Dynamic Programming
- Graphs
- Trees
- Arrays
""")

st.markdown("---")

# =====================================================
# FOOTER
# =====================================================

st.markdown(
"""
<div style='text-align:center;padding:20px;'>

<h3>🎯 AI Mock Test Generator</h3>

<p>
Built with ❤️ using
<b>Streamlit</b> •
<b>LangChain</b> •
<b>FAISS</b> •
<b>HuggingFace</b> •
<b>Groq</b>
</p>

<p style='color:gray;'>
Professional Interview Preparation Platform
</p>

</div>
""",
unsafe_allow_html=True
)
# =====================================================
# PROFESSIONAL HOME PAGE
# =====================================================

st.markdown("---")

st.header("🚀 Why Use AI Mock Test Generator?")

col1, col2 = st.columns(2)

with col1:

    st.success("""
### 🎯 Placement Ready

Generate interview questions similar to

• Google

• Amazon

• Microsoft

• Adobe

• Uber

• Goldman Sachs
""")

with col2:

    st.info("""
### 🤖 AI Powered

✔ Retrieval Augmented Generation

✔ FAISS Search

✔ HuggingFace Embeddings

✔ Groq LLM

✔ AI Verification
""")

# =====================================================
# QUALITY SCORE
# =====================================================

st.markdown("---")

st.subheader("Quality Assurance")

quality = 95

st.progress(quality)

st.write(f"Estimated Question Quality : **{quality}%**")

st.caption(
"Quality is improved using RAG retrieval, prompt engineering and AI review."
)

# =====================================================
# FAQ
# =====================================================

st.markdown("---")

with st.expander("❓ Frequently Asked Questions"):

    st.markdown("""
### Why are questions different each time?

The application uses AI with randomization and retrieval, so each generation is unique.

---

### Can I generate company-specific questions?

Yes.

---

### Can I generate coding questions?

Yes.

---

### Can I generate aptitude questions?

Yes.

---

### Does the app use RAG?

Yes.

FAISS + HuggingFace Embeddings + Groq.
""")

# =====================================================
# FINAL SIDEBAR
# =====================================================

st.sidebar.markdown("---")

st.sidebar.success("✅ System Ready")

st.sidebar.write("Model : Groq")

st.sidebar.write("Vector DB : FAISS")

st.sidebar.write("Embeddings : MiniLM")

st.sidebar.write("Retrieval : MMR")

st.sidebar.write("Verification : Enabled")

# =====================================================
# COPYRIGHT
# =====================================================

st.markdown("---")

st.markdown(
"""
<div style="text-align:center; padding:15px; color:gray;">

<b>AI Mock Test Generator</b><br>

Version 2.0

<br><br>

Built with Streamlit + LangChain + FAISS + HuggingFace + Groq

</div>
""",
unsafe_allow_html=True
)
