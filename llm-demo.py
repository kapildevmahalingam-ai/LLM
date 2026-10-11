import os
from dotenv import load_dotenv
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Load environment variables
load_dotenv()

# Page Configuration
st.set_page_config(
    page_title="Breakfast Ideas AI Assistant", page_icon="🍳", layout="centered"
)

st.title("🍳 5 Breakfast Ideas Assistant")
st.markdown(
    "Chat naturally with your 10-page breakfast guide using a RAG-powered web"
    " interface!"
)


# 2. Load and cache the new PDF vector store so it indexes only once
@st.cache_resource
def load_vector_store():
  pdf_path = (
      r"D:\SocialEagle\Program\W2_LLM_RAG\Data\5_Breakfast_Ideas_10_Page_Guide.pdf"
  )
  loader = PyPDFLoader(pdf_path)
  pages = loader.load()

  text_splitter = RecursiveCharacterTextSplitter(
      chunk_size=500, chunk_overlap=50
  )
  docs = text_splitter.split_documents(pages)

  embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
  vector_store = InMemoryVectorStore.from_documents(docs, embeddings)
  return vector_store


with st.spinner("Indexing your Breakfast Guide into the vector store..."):
  vector_store = load_vector_store()

# 3. Initialize Gemini LLM & Prompt Template for Cooking
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.3)
prompt_template = ChatPromptTemplate.from_messages([
    (
        "system",
        (
            "You are an expert culinary assistant and chef. Answer the user's"
            " question using ONLY the provided context from the breakfast guide"
            " below. If the answer cannot be found in the text, say 'I cannot"
            " find that information in the breakfast guide.'\n\nContext:\n{context}"
        ),
    ),
    ("user", "{input}"),
])
chain = prompt_template | llm

# 4. Manage Chat History in Streamlit Session State
if "messages" not in st.session_state:
  st.session_state.messages = []

# Display previous conversation messages
for message in st.session_state.messages:
  with st.chat_message(message["role"]):
    st.markdown(message["content"])

# 5. Handle User Input & Interaction Loop
if prompt := st.chat_input(
    "Ask a question about your breakfast recipes..."
):
  # Store and display user message
  st.session_state.messages.append({"role": "user", "content": prompt})
  with st.chat_message("user"):
    st.markdown(prompt)

  # Retrieve relevant context from your new PDF vector store
  retrieved_docs = vector_store.similarity_search(prompt, k=2)
  context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])

  # Generate AI response inside assistant chat container
  with st.chat_message("assistant"):
    with st.spinner("Searching breakfast guide..."):
      response = chain.invoke({"context": context_text, "input": prompt})

      # Safely extract text content
      if isinstance(response.content, list):
        rag_text = "".join(
            item.get("text", "")
            for item in response.content
            if isinstance(item, dict)
        )
      else:
        rag_text = response.content

      st.markdown(rag_text)

  # Store assistant response in chat history
  st.session_state.messages.append({"role": "assistant", "content": rag_text})