from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import getpass
import google.generativeai as genai
from typing import List
from pinecone import ServerlessSpec
from pinecone import Pinecone
import getpass

# Configure Google Gemini API
# Configure Gemini and Pinecone APIs
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not set in the environment.")
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-pro')

api_key = os.getenv("PINECONE_API_KEY")
if not api_key:
    raise ValueError("PINECONE_API_KEY is not set in the environment.")
pc = Pinecone(api_key=api_key)
index_name = "rag"  # Replace with your Pinecone index name
index = pc.Index(index_name)

# Safety settings
safe = [
    {"category": "HARM_CATEGORY_DANGEROUS", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}) 
# Function to generate embeddings
def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """Generates embeddings for a list of text chunks using Gemini."""
    embeddings = []
    for text in texts:
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=text,
            task_type="retrieval_document",
            title="PDF Chunk Embedding"
        )
        embeddings.append(result['embedding'])
    return embeddings

# Function to retrieve relevant documents (mocking Pinecone interaction)
def get_docs(query: str, top_k: int = 100) -> List[str]:
    """Retrieves relevant documents from a hypothetical document store."""
    query_embedding = generate_embeddings([query])[0]  # Generate embedding for the query
    # Mocking document retrieval (replace with actual retrieval logic if needed)
    # Replace `index.query` with actual Pinecone logic if using Pinecone.
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    return [x["metadata"]["content"] for x in results["matches"]]

# Function to generate a response using Gemini
def generate_response(query: str, docs: List[str]) -> str:
    """Generates a response using Google Gemini."""
    context = "\n---\n".join(docs)
    prompt = (
        "You are a helpful AI assistant providing answers based on my resume. If the question is not related to my resume, "
        "say 'Question is not relevant.' I have 2.5 years of experience.\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"Question: {query}"
    )
    response = model.generate_content(prompt, safety_settings=safe)
    if response is None:
        raise Exception("Failed to generate response")
    return response.text
@app.route("/")
def welcome():
    return 'welcome'
# Define API endpoint
@app.route("/generate", methods=["POST"])
def generate():
    """API endpoint to generate LLM output based on user input."""
    try:
        # Parse input data
        data = request.json
        user_query = data.get("query", "")
        
        if not user_query:
            return jsonify({"error": "Query is required."}), 400

        # Retrieve relevant documents (replace with actual logic as needed)
        retrieved_docs = get_docs(user_query)

        # Generate response
        answer = generate_response(user_query, retrieved_docs)

        # Return the response as JSON
        return jsonify({"answer": answer}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
