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

# Set up required environment variables with error handling
required_env_vars = ["MISTRAL_API_KEY", "PINECONE_API_KEY"]
missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

# Initialize the LLM (Language Model) with the system prompt
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

Always end with: "Analysis based on Supreme Court practice. Consult legal counsel for specific application.
Note: Remember to respond always in English not in Serbian language."""

llm = ChatMistralAI(model="mistral-large-latest", system_message=system_prompt)

# Initialize Pinecone for vector database
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = "us-east-1"  # Ensure this matches your Pinecone environment
pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)

# Connect to Pinecone index
index_name = "criminal-practices"
index = pc.Index(index_name)

# Initialize embedding model
embedding_function = HuggingFaceEmbeddings(
    model_name="djovak/embedic-base",
    model_kwargs={'device': 'cpu'}  # Simplified to always use CPU
)

# Add fallback message if CUDA is not available
if not torch.cuda.is_available():
    print("Warning: CUDA is not available. The model will run on CPU, which may lead to slower performance.")

# Create Pinecone vectorstore
vectorstore = PineconeVectorStore(
    index=index,
    embedding=embedding_function,
    text_key='text',
    namespace="text_chunks"
)

# Initialize retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

# Define the query refinement prompt template in English
refinement_template = """Create a focused Serbian search query for the RAG retriever bot. Convert to Serbian language if not already. Include key terms, synonyms, and domain-specific vocabulary. Remove filler words. Output only the refined query in the following format: {{refined_query}},{{keyterms}},{{synonyms}}

Query: {original_question}

Refined Query:"""

refinement_prompt = PromptTemplate(
    input_variables=["original_question"],
    template=refinement_template
)

# Create an LLMChain for query refinement using RunnableLambda
refinement_chain = refinement_prompt | llm

# Combine the system prompt with the retrieval prompt template in English
combined_template = f"""{system_prompt}

Please answer the following question using only the context provided:
{{context}}

Question: {{question}}
Answer:"""

retrieval_prompt = ChatPromptTemplate.from_template(combined_template)

# Create a retrieval chain with the combined prompt
retrieval_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": retrieval_prompt}
)

def process_query(query: str):
    try:
        # Refine the query
        refined_query_msg = refinement_chain.invoke({"original_question": query})
        
        if isinstance(refined_query_msg, dict):
            refined_query = refined_query_msg.get("text", "").strip()
        elif hasattr(refined_query_msg, 'content'):
            refined_query = refined_query_msg.content.strip()
        else:
            refined_query = str(refined_query_msg).strip()

        # Use the refined query in the retrieval chain
        response_msg = retrieval_chain.invoke(refined_query)

        # Corrected extraction of the response
        if isinstance(response_msg, dict):
            response = response_msg.get("result", "")
        elif hasattr(response_msg, 'content'):
            response = response_msg.content
        else:
            response = str(response_msg)
        
        return response
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Streamlit UI - Simplified Version
st.title("Legal Egg AI 🥚")

st.write("Welcome to Research Assistant of Serbian Supreme Court Criminal Practices! I'm an AI-powered legal assistant specializing in criminal law precedents and practice patterns from Serbia High Court. Get comprehensive analysis of case law, procedural requirements, and practical application guidelines related to Criminal Court.")

# Sidebar with example questions and clear chat button
with st.sidebar:
    st.header("Common Criminal Law Queries")
    example_questions = [
        "1. What are the latest Supreme Court positions on self-defense conditions?",
        "2. How does the Court interpret intent in corruption cases?",
        "3. What evidence standards apply in drug trafficking cases?",
        "4. Recent practice on plea bargaining requirements?",
        "5. Court's position on mandatory mitigation factors?",
        "6. How are aggravating circumstances evaluated in violent crimes?",
        "7. Standards for accepting circumstantial evidence?",
        "8. Requirements for extended confiscation?",
        "9. Practice on repeated offenses qualification?",
        "10. Court's interpretation of organized crime elements?"
    ]
    for q in example_questions:
        st.markdown(f"• {q}")
    
    st.markdown("---")
    
    # Updated button text
    if st.button("New Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
if prompt := st.chat_input("ask question..."):
    # Display user message
    with st.chat_message("user"):
        st.write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get and display assistant response
    with st.chat_message("assistant"):
        response = process_query(prompt)
        st.write(response)
    st.session_state.messages.append({"role": "assistant", "content": response})