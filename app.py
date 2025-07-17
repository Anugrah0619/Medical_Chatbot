from flask import Flask, render_template, jsonify, request
import os
from dotenv import load_dotenv
# --- RAG Specific Imports ---
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from pinecone import Pinecone 
from src.promot import system_prompt
app = Flask(__name__)

load_dotenv() # Load environment variables from .env file

# --- Retrieve API Keys ---
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

# --- Validate API Keys ---
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found in environment variables.")
if not openrouter_api_key:
    raise ValueError("OPENROUTER_API_KEY not found in environment variables.")

# --- Initialize Pinecone Client Globally ---
# This is crucial for PineconeVectorStore.from_existing_index()
try:
    Pinecone(api_key=PINECONE_API_KEY)
    print("Pinecone client initialized globally.")
except Exception as e:
    raise ConnectionError(f"Failed to initialize Pinecone client: {e}. Check your PINECONE_API_KEY.")

# --- Initialize Embeddings Model ---
print("Initializing embeddings model...")
embeddings = download_hugging_face_embeddings()
print("Embeddings model initialized.")

# --- Connect to Existing Pinecone Index ---
index_name = "medicalbot" # Ensure this matches your actual Pinecone index name
print(f"Connecting to Pinecone index: {index_name}...")
try:
    docsearch = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings
    )
    print(f"Successfully connected to Pinecone index '{index_name}'.")
except Exception as e:
    raise ConnectionError(f"Failed to connect to Pinecone index '{index_name}': {e}. Ensure the index exists and is populated.")

# --- Create Retriever ---
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})
print("Retriever created.")

# --- Initialize Language Model (LLM) ---
llm = ChatOpenAI(
    openai_api_key=openrouter_api_key,
    openai_api_base="https://openrouter.ai/api/v1",
    model_name="google/gemma-3n-e2b-it:free", # Using the free Gemma model via OpenRouter
    temperature=0.4, # Corrected syntax
    max_tokens=500
)
print(f"LLM initialized with model: {llm.model_name}")

# --- Define Chat Prompt Template ---
# Assuming system_prompt is correctly defined in src.promot
prompt = ChatPromptTemplate.from_messages(
    [
        ("human", system_prompt + "\n\nQuestion: {input}"), 
    ]
)
print("Chat prompt template created.")

# --- Create RAG Chains ---
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)
print("RAG chain created.")

# --- Flask Routes ---
@app.route("/")
def home():
    """Renders the main chat interface."""
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    """Handles chat requests and returns AI response."""
    user_message = request.json.get("message")
    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    print(f"Received message: {user_message}")
    try:
        # Invoke your RAG chain
        response = rag_chain.invoke({"input": user_message})
        ai_response = response["answer"] # LangChain's RAG chain typically returns 'answer'

        print(f"AI Response: {ai_response}")
        return jsonify({"response": ai_response})
    except Exception as e:
        print(f"Error during RAG chain invocation: {e}")
        return jsonify({"error": "Sorry, something went wrong. Please try again."}), 500

if __name__ == "__main__":
    print("Starting Flask application...")
    app.run(debug=True) # debug=True is for development, set to False in production