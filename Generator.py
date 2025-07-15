class RAGPipeline:
    def __init__(self, embedder, llm):
        self.embedder = embedder
        self.llm = llm

    def build_augmented_prompt(self, query, retrieved_chunks):
        """Builds an augmented prompt with context chunks.
        
        Args:
            query (str): User's question
            retrieved_chunks (list): List of {'content': str, 'metadata': dict}
            
        Returns:
            str: Formatted prompt with context
        """
        prompt = "Context:\n"
        for i, chunk in enumerate(retrieved_chunks, 1):
            prompt += f"Chunk {i}: {chunk['content']}\n"
        prompt += f"\nQuestion: {query}\nAnswer:"
        return prompt

    def extract_sources(self, retrieved_chunks):
        """Formats source citations from chunk metadata.
        
        Args:
            retrieved_chunks (list): Retrieved context chunks
            
        Returns:
            list: Formatted source strings
        """
        sources = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            title = chunk['metadata'].get('title', 'Untitled Document')
            link = chunk['metadata'].get('link', '#')
            sources.append(f"[{i}] {title} ({link})")
        return sources

    def run(self, query, k=3):
        """Executes full RAG pipeline.
        
        Args:
            query (str): Input question
            k (int): Number of chunks to retrieve
            
        Returns:
            tuple: (response, chunks, prompt, sources)
        """
        retrieved_chunks = self.embedder.retrieve_top_k_chunks(query, k)
        prompt = self.build_augmented_prompt(query, retrieved_chunks)
        response = self.llm.generate(prompt)
        sources = self.extract_sources(retrieved_chunks)
        return response, retrieved_chunks, prompt, sources
