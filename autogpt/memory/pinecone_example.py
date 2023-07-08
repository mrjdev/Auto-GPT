import pinecone
import numpy as np

# Initialize Pinecone
pinecone.init(api_key="fa7d752e-735c-49ef-b473-6ca455369b0e")

# Specify the index name
index_name = "vecs"

# Generate some vectors for example
num_vectors = 128
vector_dimension = 300
vectors = {str(i): np.random.rand(vector_dimension) for i in range(num_vectors)}

# Upsert vectors into the Pinecone index
pinecone.deposit(index_name, vectors)

# Query the Pinecone index
query_vector = np.random.rand(vector_dimension)
top_k_results = pinecone.fetch(index_name, query_vector, top_k=5)

# Print the results
print(top_k_results)

# Deinitialize Pinecone when done
pinecone.deinit()