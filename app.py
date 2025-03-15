import os
import torch
import streamlit as st
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_mistralai import ChatMistralAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

# Check for missing environment variables
required_env_vars = ["MISTRAL_API_KEY", "PINECONE_API_KEY"]
missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Missing environment variables: {', '.join(missing_vars)}")

# Mistral system prompt
system_prompt = """You are an expert legal assistant specializing in Serbian Supreme Court criminal practice. Your role is to provide comprehensive, practice-oriented responses that lawyers can immediately apply to their cases.

1. RESPONSE FRAMEWORK
- Start with definitive legal position based on latest practice
- Present criminal law framework: relevant Criminal Code articles + procedural rules
- List ALL applicable Supreme Court decisions chronologically
- Conclude with practical application guidelines

2. CASE LAW PRESENTATION
For each cited case:
- Full reference (Kzz/Kž number, date, panel composition)
- Key principle established
- Critical quotes from decision in Serbian
- Sentencing considerations if applicable
- Distinguishing factors from other cases
- Application requirements

3. PRACTICAL ELEMENTS
- Highlight evidence standards from precedents
- Note procedural deadlines and requirements
- Include successful defense strategies from cases
- Specify investigation requirements
- Address burden of proof patterns
- Flag prosecution weaknesses identified in similar cases

4. QUALITY CONTROLS
- Compare contradictory decisions
- Track evolution of court's interpretation
- Note recent practice changes
- Flag decisions affecting standard procedures
- Include relevant Constitutional Court positions

5. FORMATTING
- Structure: Question → Law → Cases → Application → Strategy
- Group similar precedents to show practice patterns
- Present monetary penalties in RSD (EUR)
- Use hierarchical organization for multiple precedents
- Include direct quotes for crucial legal interpretations

6. MANDATORY ELEMENTS
- Link every conclusion to specific case law
- Provide procedural guidance from precedents
- Note any practice shifts or conflicts
- Include dissenting opinions when relevant
- Reference regional court decisions confirmed by Supreme Court

Always respond in **English language only**.
"""

# Initialize Mistral LLM (with fixed system message)
llm = ChatMistralAI(model="mistral-large-latest")

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"), environment="us-east-1")
index = pc.Index("criminal-practices")

# Hugging Face embeddings for text similarity
embedding_function = HuggingFaceEmbeddings(
    model_name="djovak/embedic-base",
    model_kwargs={'device': 'cpu'} if not torch.cuda.is_available() else {'device': 'cuda'}
)

# Pinecone Vectorstore
vectorstore = PineconeVectorStore(index=index, embedding=embedding_function, text_key='text', namespace="text_chunks")

# Retriever for semantic search
retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

# Refinement Prompt
refinement_template = """Create a more focused Serbian legal search query. Include key terms, legal vocabulary, and eliminate unnecessary words. Output only the refined query:

Original Query: {original_question}

Refined Query:"""

refinement_prompt = PromptTemplate(input_variables=["original_question"], template=refinement_template)

# LLM Chain for refinement
refinement_chain = refinement_prompt | llm

# Combined Retrieval Prompt with Mistral
combined_prompt = ChatPromptTemplate.from_template(
    f"""{system_prompt}
    
    Use only the context provided below to answer the question:
    {{context}}

    Question: {{question}}
    Answer:
    """
)

# RetrievalQA Chain
retrieval_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": combined_prompt}
)

# Processing Query
def process_query(query: str):
    try:
        # Step 1: Refine Query with Mistral
        refined_query = refinement_chain.invoke({"original_question": query}).content

        # Step 2: Retrieve and Answer
        response = retrieval_chain.invoke({"query": refined_query})
        return response.get("result", "") if isinstance(response, dict) else str(response)
    except Exception as e:
        return f"Error: {str(e)}"

# Streamlit UI
st.title("Legal Egg AI 🥚")
st.write("Welcome to Research Assistant of Serbian Supreme Court Criminal Practices! I'm an AI-powered legal assistant specializing in criminal law precedents and practice patterns from Serbia High Court. Get comprehensive analysis of case law, procedural requirements, and practical application guidelines related to Criminal Court.")

# Sidebar for common legal questions
with st.sidebar:
    st.header("Common Legal Queries")
    example_questions = [
        "1. What are the latest Supreme Court positions on self-defense conditions?",
        "2. How does the Court interpret intent in corruption cases?",
        "3. What evidence standards apply in drug trafficking cases?",
        "Recent practice on plea bargaining requirements?",
        "Court's position on mitigating factors in sentencing?",
        "6. How are aggravating circumstances evaluated in violent crimes?",
        "7. Standards for accepting circumstantial evidence?",
        "8. Requirements for extended confiscation?",
    ]
    for q in example_questions:
        st.markdown(f"• {q}")

    if st.button("New Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Manage chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User input
if prompt := st.chat_input("Ask your question..."):
    with st.chat_message("user"):
        st.write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get assistant response
    with st.chat_message("assistant"):
        response = process_query(prompt)
        st.write(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
