from flask import Flask, render_template, jsonify, request
import os
import sys
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

# --- Initialize Flask App ---
app = Flask(__name__)
load_dotenv()  # Load environment variables

# --- Retrieve & Validate API Keys ---
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found in environment variables.")
if not openrouter_api_key:
    raise ValueError("OPENROUTER_API_KEY not found in environment variables.")

# --- Pinecone Initialization ---
try:
    Pinecone(api_key=PINECONE_API_KEY)
    print("Pinecone client initialized.")
except Exception as e:
    raise ConnectionError(f"Failed to initialize Pinecone: {e}")

# --- Embedding Model ---
print("Initializing embeddings...")
embeddings = download_hugging_face_embeddings()

# --- Pinecone Vector Store ---
index_name = "medicalbot"
try:
    docsearch = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings
    )
except Exception as e:
    raise ConnectionError(f"Could not connect to Pinecone index: {e}")

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# --- Language Model ---
llm = ChatOpenAI(
    openai_api_key=openrouter_api_key,
    openai_api_base="https://openrouter.ai/api/v1",
    model_name="google/gemma-3n-e2b-it:free",
    temperature=0.4,
    max_tokens=500
)

prompt = ChatPromptTemplate.from_messages([
    ("human", system_prompt + "\n\nQuestion: {input}")
])

# --- LangChain RAG Setup ---
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# --- Flask Routes ---
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message")
    if not user_message:
        return jsonify({"error": "No message provided"}), 400
    try:
        response = rag_chain.invoke({"input": user_message})
        return jsonify({"response": response["answer"]})
    except Exception as e:
        print(f"RAG Error: {e}")
        return jsonify({"error": "Something went wrong. Try again."}), 500

# --- Entry Point ---
if __name__ == "__main__":
    # Only run Flask dev server locally
    if "gunicorn" not in sys.argv[0]:
        print("Running locally with Flask server...")
        port = int(os.environ.get("PORT", 5000))
        app.run(host="0.0.0.0", port=port, debug=True)
    else:
        print("Running on Gunicorn. Flask dev server skipped.")
