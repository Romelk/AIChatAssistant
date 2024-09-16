from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from sentence_transformers import SentenceTransformer
import torch

app = FastAPI()

# Example precomputed chunks and embeddings (replace with actual embeddings)
document_chunks = [
    "A Proxy Product Owner is a representative of the Product Owner...",
    "They ensure that the development team has a clear understanding of the product requirements...",
    "The Proxy Product Owner can make decisions in the absence of the Product Owner..."
]

# Precompute embeddings using the model
model = SentenceTransformer('all-MiniLM-L6-v2')
document_embeddings = torch.tensor(model.encode(document_chunks))


# Serve the form and chat interface
@app.get("/", response_class=HTMLResponse)
async def serve_form():
    html_content = """
    <html>
        <head>
            <title>AI Chat Assistant</title>
        </head>
        <body>
            <h1>Ask a question</h1>
            <form action="/" method="post">
                <label for="question">Your question:</label>
                <input type="text" id="question" name="question">
                <input type="submit" value="Ask">
            </form>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)


# Handle the user input and return the relevant chunk
@app.post("/", response_class=HTMLResponse)
async def handle_form(question: str = Form(...)):
    # Generate the embedding for the question
    question_embedding = torch.tensor(model.encode([question]))

    # Find the most relevant chunk using cosine similarity
    closest_match = torch.cosine_similarity(document_embeddings, question_embedding, dim=1).argmax().item()
    response_chunk = document_chunks[closest_match]

    # Return the most relevant chunk to the user
    return f"<h2>The most relevant chunk is:</h2><p>{response_chunk}</p>"


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

