import torch

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


def mistral_llm(prompt, model, tokenizer):
    """Generates a response from Mistral LLM using the provided prompt.
    
    Args:
        prompt (str): The augmented prompt to generate from
        model: Loaded Mistral model
        tokenizer: Mistral tokenizer
        
    Returns:
        str: Generated answer extracted from the model output
    """
    device = 'cuda'
    print("analyzing input...")
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)
    
    with torch.no_grad():
        print("generating response...")
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False)  # deterministic for better RAG grounding

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer_start = result.find("Answer:")
    if answer_start != -1:
        result = result[answer_start + len("Answer:"):].strip()
    return result


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
