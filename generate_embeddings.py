from sentence_transformers import SentenceTransformer

def generate_embeddings(chunks):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(chunks)
    return embeddings

if __name__ == "__main__":
    # Assume `text_chunks` contains your document chunks
    text_chunks = [
        "A Proxy Product Owner is a representative of the Product Owner who acts as a liaison...",
        # other chunks
    ]
    embeddings = generate_embeddings(text_chunks)
    print("Embeddings generated: ", embeddings)

