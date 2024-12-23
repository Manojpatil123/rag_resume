from flask_lambda import FlaskLambda
from flask import request, jsonify
import os
import google.generativeai as genai
from pinecone import Pinecone

# Set up FlaskLambda instead of Flask
app = FlaskLambda(__name__)

# Configure APIs (replace with your logic)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "rag"
index = pc.Index(index_name)

# Define your routes
@app.route('/generate', methods=['POST'])
def generate():
    """Handles the /generate endpoint for LLM responses."""
    data = request.get_json()
    query = data.get('query')
    if not query:
        return jsonify({"error": "Query is required"}), 400

    # Generate embeddings and fetch documents
    query_embedding = genai.embed_content(
        model="models/text-embedding-004",
        content=query,
        task_type="retrieval_document",
    )["embedding"]
    results = index.query(vector=query_embedding, top_k=100, include_metadata=True)
    docs = [x["metadata"]["content"] for x in results["matches"]]

    # Generate response using LLM
    model = genai.GenerativeModel('gemini-1.5-pro')
    prompt = f"Context:\n{'\n---\n'.join(docs)}\n\nQuestion: {query}"
    response = model.generate_content(prompt).text

    return jsonify({"answer": response})


if __name__ == "__main__":
    app.run()
