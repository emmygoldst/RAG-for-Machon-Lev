from sentence_transformers import SentenceTransformer, util
import faiss

class Embedder:
    def __init__(self, model: str, content: list[dict]) -> None:
        '''
        Initializes the Embedder with a pre-trained model and content chunks.
        
        Args:
            model: Name/path of SentenceTransformer model (e.g., 'all-MiniLM-L6-v2')
            content: List of dictionaries containing 'content' and metadata
            
        Initializes:
            chunks: Stores the input content
            model: Loaded SentenceTransformer model
            index: FAISS index built from content embeddings
        '''
        self.chunks = content
        self.model = SentenceTransformer(model)
        self.index = self.faiss_index(self.embedding())

    def embedding(self) -> list:
        '''
        Generates embeddings for all content chunks.
        
        Returns:
            Numpy array of embeddings (shape: [num_chunks, embedding_dim])
            
        Raises:
            SystemExit: If embedding generation fails
        '''
        chunks_to_embed = [i['content'] for i in self.chunks]
        try:
            embeddings = self.model.encode(
                chunks_to_embed, 
                normalize_embeddings=True, 
                show_progress_bar=True
            )
            return embeddings
        except Exception as e:
            print(f"Error during embedding generation: {e}")
            exit(1)

    def faiss_index(self, embeddings) -> faiss.Index:
        '''
        Creates a FAISS index from embeddings.
        
        Args:
            embeddings: Numpy array of embeddings
            
        Returns:
            FAISS index object
            
        Raises:
            SystemExit: If index creation fails
        '''
        try:
            index = faiss.IndexFlatL2(embeddings.shape[1])
            index.add(embeddings)
            return index
        except Exception as e:
            print(f"Error during FAISS index creation: {e}")
            exit(1)

    def retrieve_top_k_chunks(self, query: str, k: int = 3) -> list:
        '''
        Retrieves top-k most relevant chunks for a query.
        
        Args:
            query: Search query string
            k: Number of results to return (default: 3)
            
        Returns:
            List of top-k matching content strings
        '''
        chunk_texts = [chunk['content'] for chunk in self.chunks]
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        query_embedding = query_embedding.cpu().numpy().astype('float32').reshape(1, -1)

        distances, indices = self.index.search(query_embedding, k)
        return [chunk_texts[i] for i in indices[0]]
