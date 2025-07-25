from bert_score import score
import json
from Generator import *

class RAGEvaluator:
    def __init__(self, embedder, llm):
        self.embedder = embedder
        self.llm = llm
    def bert_score_eval(self, reference, candidate):
        """Calculate BERTScore metrics between reference and candidate text.
        
        Args:
            reference (str): Ground truth text
            candidate (str): Generated text to evaluate
            
        Returns:
            dict: Precision, Recall, and F1 scores
        """
        P, R, F1 = score([candidate], [reference], lang="en", rescale_with_baseline=True)
        return {
            "Precision": round(P[0].item(), 3),
            "Recall": round(R[0].item(), 3),
            "F1": round(F1[0].item(), 3)
        }

    def run_batch_eval(self, file_path):
        """Run batch evaluation on a set of QA pairs.
        
        Args:
            file_path (str): Path to JSON file with questions/references
            rag_pipeline_func (callable): Function that takes a question and returns RAG output
            
        Returns:
            list: Evaluation results for each question
        """
        with open(file_path, 'r') as f:
            question_set = json.load(f)

        results = []

        for idx, item in enumerate(question_set, 1):
            question = item["question"]
            reference = item["reference_answer"]

            print(f"\n{'=' * 80}")
            print(f"Question {idx}:\n{question}")
            print(f"\nReference Answer:\n{reference}")

            # Run RAG pipeline
            response, retrieved_chunks, prompt, sources = run_rag_pipeline(self.embedder, self.llm, question, k=4)

            print(f"\nModel Answer:\n{response}")

            # Print retrieved context
            print("\nRetrieved Context Chunks:")
            for i, chunk in enumerate(retrieved_chunks, 1):
                print(f"\nChunk {i}:\n{chunk['content']}")
                print(f"(Source: {chunk['metadata'].get('title', '')})")
                print("-" * 40)

            # Automatic evaluation
            scores = self.bert_score_eval(reference, response)
            print(f"\nBERTScore Evaluation:")
            print(f"Precision: {scores['Precision']}, Recall: {scores['Recall']}, F1: {scores['F1']}")

            # Save results
            results.append({
                "question": question,
                "reference_answer": reference,
                "model_answer": response,
                "bertscore_precision": scores["Precision"],
                "bertscore_recall": scores["Recall"],
                "bertscore_f1": scores["F1"],
                "retrieved_chunks": [chunk["content"] for chunk in retrieved_chunks],
                "sources": sources
            })

        return results
