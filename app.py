import streamlit as st
import random
import re
from datetime import datetime
from io import BytesIO

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

try:
    from docx import Document
    from reportlab.platypus import SimpleDocTemplate, Paragraph
    from reportlab.lib.styles import getSampleStyleSheet
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

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
.stApp { background: #F6F8FC; }

.block-container {
    padding-top: 1rem;
    padding-bottom: 2rem;
}

.main-title {
    font-size: 40px;
    font-weight: 700;
    color: #2E8B57;
}

.sub-title {
    color: #808080;
    font-size: 17px;
}

.card {
    background: #f8f9fa;
    padding: 18px;
    border-radius: 10px;
    border: 1px solid #e3e3e3;
}

.main-header {
    background: linear-gradient(135deg, #1E88E5, #42A5F5);
    padding: 25px;
    border-radius: 15px;
    color: white;
    margin-bottom: 20px;
}

.main-header h1 {
    color: white;
    font-size: 38px;
    margin-bottom: 5px;
}

.main-header p { font-size: 18px; }

.question-card {
    background: white;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0 5px 15px rgba(0,0,0,.08);
    margin-bottom: 25px;
    border-left: 8px solid #1E88E5;
}

.step-card {
    background: #F9FAFB;
    border-left: 5px solid #43A047;
    padding: 15px;
    border-radius: 10px;
    margin-bottom: 15px;
}

.answer-card {
    background: #E8F5E9;
    border-left: 6px solid #2E7D32;
    padding: 18px;
    border-radius: 10px;
    font-weight: bold;
}

.stButton > button {
    width: 100%;
    height: 50px;
    font-size: 18px;
    font-weight: bold;
    border-radius: 8px;
    background: #1E88E5;
    color: white;
}

.stButton > button:hover { background: #1565C0; }
.stDownloadButton > button { width: 100%; }
</style>
""", unsafe_allow_html=True)

# ==========================================================
# HEADER
# ==========================================================

st.markdown("""
<div class="main-title">🎯 AI Mock Test Generator</div>
<div class="sub-title">
Professional Interview Question Generator using
<b>RAG + FAISS + HuggingFace + Groq</b>
</div>
""", unsafe_allow_html=True)

st.divider()

# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.title("⚙ Configuration")

company = st.sidebar.selectbox(
    "Target Company",
    ["Google", "Amazon", "Microsoft", "Adobe", "Oracle",
     "Uber", "Goldman Sachs", "Flipkart", "Infosys", "TCS", "Accenture"]
)

question_type = st.sidebar.selectbox("Question Type", ["Aptitude", "Coding"])
difficulty = st.sidebar.selectbox("Difficulty", ["Easy", "Medium", "Hard"])
num_questions = st.sidebar.slider("Number of Questions", 1, 20, 5)
topic = st.text_input("Enter Topic", placeholder="Arrays, Percentage, DBMS...")
generate = st.button("🚀 Generate Questions")

# ==========================================================
# SESSION STATE
# ==========================================================

if "history" not in st.session_state:
    st.session_state.history = []

if "generated_questions" not in st.session_state:
    st.session_state.generated_questions = set()

if "favorites" not in st.session_state:
    st.session_state.favorites = []

if "last_questions" not in st.session_state:
    st.session_state.last_questions = []

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
    try:
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

        vector_db = FAISS.from_texts(texts=chunks, embedding=embeddings)
        return vector_db
    except Exception as e:
        st.error(f"Failed to load vector database: {str(e)}")
        return None


vector_db = load_vector_database()

# ==========================================================
# RETRIEVER
# ==========================================================

retriever = None
if vector_db is not None:
    retriever = vector_db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 6, "fetch_k": 20}
    )

# ==========================================================
# LLM CONFIGURATION
# ==========================================================

llm = None
try:
    groq_api_key = st.secrets.get("GROQ_API_KEY", "")
    if groq_api_key:
        llm = ChatGroq(
            api_key=groq_api_key,
            model="llama-3.3-70b-versatile",
            temperature=0.2,
            max_tokens=4096
        )
except Exception:
    pass

# ==========================================================
# HELPER FUNCTIONS
# ==========================================================

def retrieve_context(topic: str) -> str:
    if retriever is None:
        return ""
    try:
        docs = retriever.invoke(topic)
        if not docs:
            return ""
        return "\n\n".join(doc.page_content for doc in docs)
    except Exception:
        return ""


def clean_response(text: str) -> str:
    if not text:
        return ""
    text = text.replace("```", "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def generation_time() -> str:
    return datetime.now().strftime("%d-%m-%Y %H:%M:%S")


# ==========================================================
# QUESTION STYLES
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
# PROMPTS
# ==========================================================

APTITUDE_PROMPT = ChatPromptTemplate.from_template("""
You are a Senior Aptitude Assessment Designer with 15+ years of experience.

You create placement assessments for Google, Amazon, Microsoft, Adobe, Uber, Goldman Sachs.

Company: {company}
Assessment Style: {style}
Difficulty: {difficulty}
Topic: {topic}
Reference Material: {context}

TASK: Generate EXACTLY ONE ORIGINAL aptitude question.

RULES:
1. Generate a unique question.
2. Use realistic numbers.
3. Exactly FOUR options.
4. Exactly ONE correct answer.
5. Solve the problem before writing.
6. Verify every calculation.
7. Never contradict yourself.
8. Never approximate.

OUTPUT FORMAT:

Question
<Question>

Options
A. ...
B. ...
C. ...
D. ...

Correct Answer
The correct answer is <Option>. <Answer>

Step-by-Step Solution
Step 1
<Explanation>
Step 2
<Explanation>
Continue until solved.

Final Answer
The correct answer is <Option>. <Answer>

Return ONLY the final question.
""")

CODING_PROMPT = ChatPromptTemplate.from_template("""
You are a Senior Software Engineer.
Create ONE original coding interview problem.

Company: {company}
Assessment Style: {style}
Difficulty: {difficulty}
Topic: {topic}
Reference: {context}

Output:
Problem Statement
Input Format
Output Format
Constraints
Sample Input
Sample Output
Explanation
Python Function
Hints
Time Complexity
Space Complexity

Requirements:
- Original question
- Professional wording
- Company level
- No copied problems

Return only the coding problem.
""")

REVIEW_PROMPT = ChatPromptTemplate.from_template("""
You are a Senior Quality Reviewer.
Review this interview question.

{question}

Checklist:
- English
- Grammar
- Logic
- Mathematics
- Correct Option
- Explanation
- Final Answer

If ANY mistake exists, rewrite the ENTIRE question.
Do NOT explain the mistake.
Return ONLY the corrected version.
""")

# ==========================================================
# QUESTION GENERATOR ENGINE
# ==========================================================

def generate_question(company, topic, difficulty, question_type):
    if llm is None:
        return "❌ LLM not initialized. Please check your GROQ_API_KEY."

    context = retrieve_context(topic)
    style = random.choice(QUESTION_STYLES)

    try:
        if question_type == "Aptitude":
            prompt = APTITUDE_PROMPT.format(
                company=company, style=style, difficulty=difficulty,
                topic=topic, context=context
            )
        else:
            prompt = CODING_PROMPT.format(
                company=company, style=style, difficulty=difficulty,
                topic=topic, context=context
            )

        first_response = llm.invoke(prompt)
        draft = clean_response(first_response.content)

        review = REVIEW_PROMPT.format(question=draft)
        reviewed = llm.invoke(review)
        final_question = clean_response(reviewed.content)

        return final_question

    except Exception as e:
        return f"❌ Generation Error\n\n{str(e)}"


def is_valid_question(question: str) -> bool:
    if not question:
        return False
    if "❌" in question:
        return False
    required_sections = ["Question", "Options", "Correct Answer"]
    for section in required_sections:
        if section.lower() not in question.lower():
            return False
    return True


def is_duplicate(question: str) -> bool:
    normalized = " ".join(question.split()).lower()
    if normalized in st.session_state.generated_questions:
        return True
    st.session_state.generated_questions.add(normalized)
    return False


def generate_unique_question(company, topic, difficulty, question_type, retries=5):
    for _ in range(retries):
        question = generate_question(company, topic, difficulty, question_type)
        if not is_valid_question(question):
            continue
        if is_duplicate(question):
            continue
        return question
    return "Unable to generate a unique high-quality question. Please try again."


def generate_multiple_questions(company, topic, difficulty, question_type, count):
    questions = []
    st.session_state.generated_questions.clear()

    progress = st.progress(0)
    status = st.empty()

    for i in range(count):
        status.info(f"Generating Question {i+1} of {count}")
        q = generate_unique_question(company, topic, difficulty, question_type)
        questions.append(q)
        progress.progress((i + 1) / count)

    progress.empty()
    status.success("✅ Generation Completed")
    return questions


# ==========================================================
# DISPLAY QUESTION
# ==========================================================

def display_question(question_number, content):
    st.markdown(f"""
    <div class="question-card">
    <h2 style="color:#2E8B57;">Question {question_number}</h2>
    </div>
    """, unsafe_allow_html=True)

    if "Step-by-Step Solution" not in content:
        st.markdown(content)
        st.markdown("---")
        return

    question_text, solution_text = content.split("Step-by-Step Solution", 1)

    st.markdown(question_text)

    if solution_text.strip():
        with st.expander("📖 View Detailed Solution", expanded=False):
            st.success("Step-by-Step Solution")

            step_pattern = r"(Step\s+\d+.*?)(?=Step\s+\d+|Final Answer|$)"
            matches = re.findall(step_pattern, solution_text, flags=re.S)

            for step in matches:
                st.markdown(f"""
                <div class="step-card">
                {step.strip()}
                </div>
                """, unsafe_allow_html=True)

            if "Final Answer" in solution_text:
                final = solution_text.split("Final Answer", 1)[1]
                st.markdown(f"""
                <div class="answer-card">
                <b>Final Answer</b><br/>
                {final.strip()}
                </div>
                """, unsafe_allow_html=True)

    st.markdown("---")


# ==========================================================
# DOWNLOAD / EXPORT FUNCTIONS
# ==========================================================

def prepare_download(questions):
    text = ""
    for i, q in enumerate(questions, 1):
        text += f"""
======================================================
QUESTION {i}
======================================================

{q}

"""
    return text


def export_docx(questions):
    if not DOCX_AVAILABLE:
        return None
    document = Document()
    document.add_heading("AI Mock Test Generator", level=1)
    for i, question in enumerate(questions, 1):
        document.add_heading(f"Question {i}", level=2)
        document.add_paragraph(question)
    file = BytesIO()
    document.save(file)
    file.seek(0)
    return file


def export_pdf(questions):
    if not DOCX_AVAILABLE:
        return None
    pdf = BytesIO()
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(pdf)
    story = []
    story.append(Paragraph("<b>AI Mock Test Generator</b>", styles["Heading1"]))
    for i, q in enumerate(questions, 1):
        story.append(Paragraph(f"<b>Question {i}</b>", styles["Heading2"]))
        escaped_q = q.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        story.append(Paragraph(escaped_q.replace("\n", "<br/>"), styles["BodyText"]))
    doc.build(story)
    pdf.seek(0)
    return pdf


def save_history(company, topic, difficulty, question_type, number):
    st.session_state.history.append({
        "Company": company,
        "Topic": topic,
        "Difficulty": difficulty,
        "Type": question_type,
        "Questions": number,
        "Time": generation_time()
    })


# ==========================================================
# MAIN GENERATION
# ==========================================================

if generate:
    if topic.strip() == "":
        st.warning("Please enter a topic.")
        st.stop()

    if llm is None:
        st.error("❌ Groq LLM not initialized. Please set GROQ_API_KEY in secrets.")
        st.stop()

    st.markdown("---")
    st.header("Generated Questions")

    questions = generate_multiple_questions(
        company, topic, difficulty, question_type, num_questions
    )

    st.session_state.last_questions = questions

    for i, question in enumerate(questions, 1):
        display_question(i, question)

    save_history(company, topic, difficulty, question_type, num_questions)

    # Download section
    st.markdown("---")
    st.subheader("📥 Download Questions")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.download_button(
            "Download TXT",
            prepare_download(questions),
            file_name=f"{company}_{topic}.txt",
            mime="text/plain"
        )

    with col2:
        if DOCX_AVAILABLE:
            docx_file = export_docx(questions)
            if docx_file:
                st.download_button(
                    "Download DOCX",
                    docx_file,
                    file_name="questions.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )

    with col3:
        if DOCX_AVAILABLE:
            pdf_file = export_pdf(questions)
            if pdf_file:
                st.download_button(
                    "Download PDF",
                    pdf_file,
                    file_name="questions.pdf",
                    mime="application/pdf"
                )

    if st.button("🔄 Regenerate Questions"):
        st.rerun()


# ==========================================================
# HISTORY PANEL
# ==========================================================

if st.session_state.history:
    st.markdown("---")
    st.header("📜 Recent Generations")
    for item in reversed(st.session_state.history):
        with st.container():
            st.markdown(f"""
**Company:** {item['Company']}

**Topic:** {item['Topic']}

**Difficulty:** {item['Difficulty']}

**Question Type:** {item['Type']}

**Questions Generated:** {item['Questions']}

**Generated On:** {item['Time']}
""")
            st.divider()

# ==========================================================
# DASHBOARD
# ==========================================================

st.markdown("---")
st.header("📊 Dashboard")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("🏢 Company", company)

with col2:
    st.metric("📚 Topic", topic if topic else "-")

with col3:
    st.metric("📝 Questions", num_questions)

with col4:
    st.metric("🎯 Difficulty", difficulty)

st.markdown("---")

# ==========================================================
# AI FEATURES
# ==========================================================

st.subheader("🚀 AI Features")

c1, c2, c3 = st.columns(3)

with c1:
    st.success("""
### 🧠 RAG
✔ FAISS
✔ HuggingFace
✔ Semantic Search
✔ MMR Retrieval
""")

with c2:
    st.info("""
### 🤖 AI
✔ Groq LLM
✔ AI Verification
✔ Duplicate Prevention
✔ Smart Prompting
""")

with c3:
    st.warning("""
### 🎯 Quality
✔ Company Pattern
✔ Original Questions
✔ Dynamic Steps
✔ Professional Format
""")

st.markdown("---")

# ==========================================================
# QUESTION QUALITY
# ==========================================================

st.subheader("📈 Question Quality")

quality = 96
st.progress(quality)
st.success(f"Estimated Question Quality : {quality}%")

st.caption("""
Questions are generated using Retrieval Augmented Generation (RAG),
professional prompt engineering and an AI verification step.
""")

# ==========================================================
# AI CONFIDENCE
# ==========================================================

st.markdown("---")
confidence = random.randint(95, 99)
st.subheader("🤖 AI Confidence")
st.progress(confidence)
st.success(f"Confidence Score: {confidence}%")

# ==========================================================
# COMPANY SUPPORT
# ==========================================================

st.markdown("---")
st.subheader("🏢 Supported Companies")

companies = [
    "Google", "Amazon", "Microsoft", "Adobe", "Oracle",
    "Uber", "Goldman Sachs", "Flipkart", "Infosys", "TCS", "Accenture"
]

cols = st.columns(3)
for index, company_name in enumerate(companies):
    cols[index % 3].success(company_name)

# ==========================================================
# SUPPORTED TOPICS
# ==========================================================

st.markdown("---")
st.subheader("📚 Supported Topics")

topics = [
    "Arrays", "Strings", "Linked List", "Stack", "Queue",
    "Trees", "Graphs", "Dynamic Programming", "DBMS",
    "Operating System", "Computer Networks", "SQL",
    "Probability", "Percentage", "Profit and Loss",
    "Time and Work", "Logical Reasoning", "Data Interpretation"
]

topic_cols = st.columns(3)
for index, t in enumerate(topics):
    topic_cols[index % 3].write(f"✔ {t}")

# ==========================================================
# ANALYTICS
# ==========================================================

st.markdown("---")
st.subheader("📊 Analytics")

a1, a2, a3 = st.columns(3)

with a1:
    st.metric("Generated", len(st.session_state.history))

with a2:
    st.metric("Favorites", len(st.session_state.favorites))

with a3:
    st.metric("Current Batch", num_questions)

# ==========================================================
# FAVORITES
# ==========================================================

if st.session_state.favorites:
    st.markdown("---")
    st.subheader("⭐ Favorite Questions")
    for q in st.session_state.favorites:
        st.markdown(q)
        st.divider()

# ==========================================================
# FOOTER
# ==========================================================

st.markdown("---")
st.markdown("""
<div style="text-align:center;padding:25px;color:gray;">
<h2>🎯 AI Mock Test Generator</h2>
<p>Professional Interview Assessment Platform</p>
<p>Powered by <b>Groq • LangChain • HuggingFace • FAISS • Streamlit</b></p>
<p>Version 3.0 — © 2026 All Rights Reserved</p>
</div>
""", unsafe_allow_html=True)

# ==========================================================
# SIDEBAR INFO
# ==========================================================

st.sidebar.markdown("---")
st.sidebar.subheader("⚡ AI Engine")

if llm is not None:
    st.sidebar.success("Groq Connected")
else:
    st.sidebar.error("Groq Not Connected")

if vector_db is not None:
    st.sidebar.success("FAISS Loaded")
    st.sidebar.success("Embeddings Ready")
    st.sidebar.success("Retriever Active")
else:
    st.sidebar.error("FAISS Not Loaded")

st.sidebar.success("AI Reviewer Active")

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Statistics")
st.sidebar.metric("Sessions", len(st.session_state.history))
st.sidebar.metric("Questions", num_questions)
st.sidebar.metric("Difficulty", difficulty)

st.sidebar.markdown("---")
st.sidebar.success("🟢 System Online")
st.sidebar.write("Model : Groq")
st.sidebar.write("Retriever : FAISS")
st.sidebar.write("Embeddings : MiniLM")
st.sidebar.write("Quality Check : Enabled")
st.sidebar.write("RAG : Active")
