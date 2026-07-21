import streamlit as st
import os
import tempfile
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------

st.set_page_config(
    page_title="AI Mock Test Generator",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 AI Mock Test Generator")
st.markdown("Generate Aptitude & Coding Questions using **RAG + Groq**")

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------

st.sidebar.header("Configuration")

company = st.sidebar.text_input(
    "Company Name",
    value="Google"
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
    1
)

uploaded_file = st.file_uploader(
    "Upload Study Material (.txt)",
    type=["txt"]
)

topic = st.text_input(
    "Enter Topic",
    placeholder="Arrays, Dynamic Programming, DBMS..."
)

generate = st.button("Generate Questions")

# -------------------------------------------------
# VECTOR DATABASE
# -------------------------------------------------

@st.cache_resource
def build_vector_store(text):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_text(text)

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    db = FAISS.from_texts(
        chunks,
        embeddings
    )

    return db


def retrieve_context(db, query):

    retriever = db.as_retriever(
        search_kwargs={"k":3}
    )

    docs = retriever.invoke(query)

    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    return context

# -------------------------------------------------
# GROQ MODEL
# -------------------------------------------------

llm = ChatGroq(
    api_key=st.secrets["GROQ_API_KEY"],
    model="llama-3.1-8b-instant",
    temperature=0.7
)


# -------------------------------------------------
# PROMPT TEMPLATES
# -------------------------------------------------

aptitude_prompt = ChatPromptTemplate.from_template("""
You are an expert aptitude interviewer for {company}.

Use ONLY the given context.

Context:
{context}

Generate ONE {difficulty} level aptitude question on:

Topic: {topic}

Requirements:

1. Multiple Choice Question
2. Four options (A, B, C, D)
3. Correct Answer
4. Detailed Explanation
5. Similar to {company} placement assessment.
""")

coding_prompt = ChatPromptTemplate.from_template("""
You are an expert coding interviewer for {company}.

Use ONLY the given context.

Context:
{context}

Generate ONE {difficulty} coding interview problem.

Topic:
{topic}

Requirements:

1. Problem Statement
2. Constraints
3. Input Format
4. Output Format
5. Sample Input
6. Sample Output
7. Hidden Test Cases
8. Python Function Signature
9. Difficulty
10. Time Complexity
11. Space Complexity
12. Hints
""")

# -------------------------------------------------
# GENERATION FUNCTION
# -------------------------------------------------

def generate_question(
    context,
    company,
    topic,
    difficulty,
    question_type
):

    if question_type == "Aptitude":

        prompt = aptitude_prompt.format(
            company=company,
            context=context,
            topic=topic,
            difficulty=difficulty
        )

    else:

        prompt = coding_prompt.format(
            company=company,
            context=context,
            topic=topic,
            difficulty=difficulty
        )

    response = llm.invoke(prompt)

    return response.content

# -------------------------------------------------
# GENERATE QUESTIONS
# -------------------------------------------------

if generate:

    if uploaded_file is None:
        st.error("Please upload a study material (.txt) file.")
        st.stop()

    if topic.strip() == "":
        st.error("Please enter a topic.")
        st.stop()

    with st.spinner("Reading study material..."):

        text = uploaded_file.read().decode("utf-8")

    with st.spinner("Building Vector Database..."):

        db = build_vector_store(text)

    with st.spinner("Retrieving Context..."):

        context = retrieve_context(db, topic)

    st.success("Knowledge Base Ready!")

    st.divider()

    st.subheader("Generated Questions")

    progress = st.progress(0)

    output_text = ""

    for i in range(1, num_questions + 1):

        progress.progress(i / num_questions)

        with st.spinner(f"Generating Question {i}..."):

            result = generate_question(
                context=context,
                company=company,
                topic=topic,
                difficulty=difficulty,
                question_type=question_type
            )

        st.markdown(f"## Question {i}")

        st.markdown(result)

        st.divider()

        output_text += f"\n\n========== QUESTION {i} ==========\n\n"
        output_text += result
        output_text += "\n\n"

    st.success("All Questions Generated Successfully!")

    st.download_button(
        label="📥 Download Questions",
        data=output_text,
        file_name=f"{company}_{topic}_{question_type}.txt",
        mime="text/plain"
    )

# -------------------------------------------------
# SIDEBAR INFO
# -------------------------------------------------

st.sidebar.markdown("---")

st.sidebar.info(
    """
### 📌 Instructions

1. Upload a **.txt** study material file.

2. Enter the topic.

3. Select company.

4. Select question type.

5. Choose difficulty.

6. Choose number of questions.

7. Click **Generate Questions**.
"""
)

# -------------------------------------------------
# CONTEXT PREVIEW
# -------------------------------------------------

if uploaded_file is not None:

    st.divider()

    st.subheader("📖 Uploaded Study Material Preview")

    uploaded_file.seek(0)

    preview_text = uploaded_file.read().decode("utf-8")

    st.text_area(
        "Preview",
        preview_text[:3000],
        height=250
    )

# -------------------------------------------------
# ABOUT
# -------------------------------------------------

with st.expander("ℹ About this Application"):

    st.markdown(
        """
### AI Mock Test Generator

This application uses:

- ✅ Streamlit
- ✅ LangChain
- ✅ HuggingFace Embeddings
- ✅ FAISS Vector Database
- ✅ Groq LLM
- ✅ Retrieval Augmented Generation (RAG)

Generate company-specific:

- Aptitude Questions
- Coding Questions

Supported Companies:

- Google
- Amazon
- Microsoft
- Adobe
- Oracle
- TCS
- Infosys
- Accenture
- Deloitte
- Capgemini
- Cognizant

Upload your own study material and generate unlimited interview questions.
"""
    )

# -------------------------------------------------
# FOOTER
# -------------------------------------------------

st.divider()

st.markdown(
"""
<center>

Made with ❤️ using

**Streamlit • LangChain • FAISS • HuggingFace • Groq**

</center>
""",
unsafe_allow_html=True
)
