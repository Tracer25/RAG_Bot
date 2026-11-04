def retrieve_documents(query, index, documents, model):
    query_embedding = model.encode([query])

    distances, indices = index.search(query_embedding, k=5)

    return [documents[i] for i in indices[0]]
