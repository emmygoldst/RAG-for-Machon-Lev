def build_augmented_prompt(query, retrieved_chunks):
      """Builds an augmented prompt for RAG by combining retrieved chunks with the query.
      
      Args:
          query (str): The user's question/query
          retrieved_chunks (list): List of dictionary chunks with 'content' and 'metadata' keys
          
      Returns:
          str: Formatted prompt with context and question
      """
      prompt = "Context:\n"
      for i, chunk in enumerate(retrieved_chunks, 1):
          prompt += f"Chunk {i}: {chunk['content']}\n"
      prompt += f"Question: {query}\nAnswer:"
      return prompt
  
def extract_sources(retrieved_chunks):
      """Extracts source information from retrieved chunks for citation.
      
      Args:
          retrieved_chunks (list): List of dictionary chunks with 'metadata' keys
          
      Returns:
          list: Formatted source strings with titles and links
      """
      sources = []
      for i, chunk in enumerate(retrieved_chunks, 1):
          title = chunk['metadata'].get('title', 'Unknown Title')
          link = chunk['metadata'].get('link', 'No Link')
          sources.append(f"[{i}] {title} - {link}")
      return sources
      
def run_rag_pipeline(embedder, llm, query, k=3):
      """ Executes the RAG pipeline for a given query.

      Args:
          embedder (Embedder): Instance of the Embedder class
          llm (LLM): Instance of the LLM class
          query (str): The user's question/query
          k (int): Number of chunks to retrieve

      Returns:
          tuple: (response, retrieved_chunks, prompt, sources)
      """
      retrieved_chunks = embedder.retrieve_top_k_chunks(query, k)
      prompt = build_augmented_prompt(query, retrieved_chunks)
      response = llm.mistral_llm(prompt)
      sources = extract_sources(retrieved_chunks)
      return response, retrieved_chunks, prompt, sources
