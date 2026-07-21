import streamlit as st
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(
    page_title="AI Mock Test Generator",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 AI Mock Test Generator")
st.write("Generate Aptitude and Coding Questions using RAG + Groq")

# -------------------------
# SIDEBAR
# -------------------------
st.sidebar.header("Configuration")

groq_api_key = st.sidebar.text_input(
    "Groq API Key",
    type="password"
)

company_name = st.sidebar.text_input(
    "Company Name",
    value="Google"
)

question_type = st.sidebar.selectbox(
    "Question Type",
    ["Aptitude", "Coding"]
)

uploaded_file = st.file_uploader(
    "Upload Study Material (.txt)",
    type=["txt"]
)

topic = st.text_input(
    "Enter Topic",
    placeholder="Dynamic Programming"
)

# -------------------------
# VECTOR STORE
# -------------------------
@st.cache_resource
def create_vector_store(text):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_text(text)

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    vector_store = FAISS.from_texts(
        chunks,
        embeddings
    )

    return vector_store


def retrieve_context(retriever, query):

    docs = retriever.invoke(query)

    return "\n\n".join(
        doc.page_content
        for doc in docs
    )


# -------------------------
# PROMPTS
# -------------------------
aptitude_prompt = ChatPromptTemplate.from_template("""
You are an expert aptitude interviewer for {company}.

Use the provided context to generate exactly ONE aptitude question.

Topic:
{topic}

Context:
{context}

Requirements:
1. Multiple choice question.
2. Four options (A,B,C,D).
3. Clearly mention correct answer.
4. Provide explanation.
5. Question should resemble real company assessments.
""")


coding_prompt = ChatPromptTemplate.from_template("""
You are an expert coding interviewer for {company}.

Use the provided context to generate exactly ONE coding question.

Topic:
{topic}

Context:
{context}

Requirements:
1. Problem statement.
2. Input format.
3. Output format.
4. Constraints.
5. Two sample test cases.
6. Python solution template.
7. Difficulty level.
""")


# -------------------------
# GENERATE
# -------------------------
if st.button("Generate Question"):

    if not groq_api_key:
        st.error("Enter Groq API Key")
        st.stop()

    if uploaded_file is None:
        st.error("Upload study material")
        st.stop()

    if not topic:
        st.error("Enter topic")
        st.stop()

    text = uploaded_file.read().decode("utf-8")

    with st.spinner("Creating Vector Database..."):
        vector_store = create_vector_store(text)
        retriever = vector_store.as_retriever(
            search_kwargs={"k": 3}
        )

    context = retrieve_context(
        retriever,
        topic
    )

    llm = ChatGroq(
        api_key=groq_api_key,
        model="llama-3.1-8b-instant",
        temperature=0.7
    )

    if question_type == "Aptitude":

        prompt = aptitude_prompt.format(
            company=company_name,
            topic=topic,
            context=context
        )

    else:

        prompt = coding_prompt.format(
            company=company_name,
            topic=topic,
            context=context
        )

    with st.spinner("Generating Question..."):
        response = llm.invoke(prompt)

    st.success("Question Generated Successfully")

    st.markdown(response.content)
