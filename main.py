import time
import textwrap  # Import the textwrap module
from  Evaluation import RAGEvaluator
from Generator import *
def main(embedder, llm):
    mode = input("Choose mode: [1] Single question | [2] Evaluate file: ")

    if mode.strip() == "2":
        eval_file = input("Enter path to eval file (default: eval_questions.json): ").strip()
        if not eval_file:
            eval_file = "eval_questions.json"
        evaluator = RAGEvaluator(embedder, llm)
        results = evaluator.run_batch_eval(eval_file)
        return

    while True:
        query = input("Please enter your question (or type 'quit' to exit): ")

        if query.lower() == 'quit':
            print("Exiting program.")
            break
        if not query.strip():
            print("Please enter a valid question.")
            continue

        print("Processing your question...")
        start_time = time.time()
        try:
            response, retrieved_chunks, prompt, sources = run_rag_pipeline(embedder, llm, query, k=3)
            # Wrap the response text
            wrapped_response = textwrap.fill(response, width=80) # You can adjust the width here
            print(f"Answer:\n{wrapped_response}")
            print("Sources:")
            for source in sources:
                print(source)
            chunk = input("Print chunks? y/n: ")
            if chunk.strip() == "y":
                print("Retrieved chunks:")
                for i, chunk in enumerate(retrieved_chunks, 1):
                    print(f"Chunk {i}:\n{chunk['content']}")
                    print(f"(Source: {chunk['metadata'].get('title', '')})\n{'-' * 50}")
            end_time = time.time()
            print(f"Time taken: {end_time - start_time:.2f} seconds")
            print("-" * 60)
        except Exception as e:
            print(f"An error occurred during the RAG pipeline: {e}")
