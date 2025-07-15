from bert_score import score
import json
from typing import List, Dict, Optional
from rouge_score import rouge_scorer
from Generator import RAGPipeline  # Assuming Generator.py contains RAGPipeline class

class RAGEvaluator:
    """Handles evaluation of RAG pipeline outputs against reference answers"""
    
    def __init__(self, rag_pipeline: RAGPipeline):
        """
        Args:
            rag_pipeline: Initialized RAG pipeline instance
        """
        self.rag = rag_pipeline
        self.rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    def _calculate_bertscore(self, reference: str, candidate: str) -> Dict[str, float]:
        """Compute BERTScore metrics"""
        P, R, F1 = score([candidate], [reference], lang="en", rescale_with_baseline=True)
        return {
            "precision": round(P[0].item(), 3),
            "recall": round(R[0].item(), 3),
            "f1": round(F1[0].item(), 3)
        }

    def _calculate_rouge(self, reference: str, candidate: str) -> Dict[str, Dict[str, float]]:
        """Compute ROUGE metrics"""
        scores = self.rouge.score(reference, candidate)
        return {
            "rouge1": {
                "precision": round(scores['rouge1'].precision, 3),
                "recall": round(scores['rouge1'].recall, 3),
                "f1": round(scores['rouge1'].fmeasure, 3)
            },
            "rouge2": {
                "precision": round(scores['rouge2'].precision, 3),
                "recall": round(scores['rouge2'].recall, 3),
                "f1": round(scores['rouge2'].fmeasure, 3)
            },
            "rougeL": {
                "precision": round(scores['rougeL'].precision, 3),
                "recall": round(scores['rougeL'].recall, 3),
                "f1": round(scores['rougeL'].fmeasure, 3)
            }
        }

    def evaluate_single(self, question: str, reference_answer: str, verbose: bool = True) -> Dict:
        """Evaluate single QA pair"""
        try:
            # Run RAG pipeline
            response, chunks, _, sources = self.rag.run(question)
            
            # Calculate metrics
            bert_scores = self._calculate_bertscore(reference_answer, response)
            rouge_scores = self._calculate_rouge(reference_answer, response)
            
            if verbose:
                print(f"\n{'=' * 80}")
                print(f"Question: {question}")
                print(f"\nReference Answer:\n{reference_answer}")
                print(f"\nModel Answer:\n{response}")
                
                print("\nRetrieved Context Chunks:")
                for i, chunk in enumerate(chunks, 1):
                    print(f"\nChunk {i}:\n{chunk['content']}")
                    print(f"(Source: {chunk['metadata'].get('title', '')})")
                    print("-" * 40)
                
                print("\nEvaluation Scores:")
                print(f"BERTScore F1: {bert_scores['f1']}")
                print(f"ROUGE-L F1: {rouge_scores['rougeL']['f1']}")

            return {
                "question": question,
                "reference_answer": reference_answer,
                "model_answer": response,
                "metrics": {
                    "bertscore": bert_scores,
                    "rouge": rouge_scores
                },
                "retrieved_chunks": [chunk["content"] for chunk in chunks],
                "sources": sources
            }
            
        except Exception as e:
            print(f"Error evaluating question '{question}': {str(e)}")
            return None

    def evaluate_batch(self, file_path: str, verbose: bool = False) -> List[Dict]:
        """Evaluate all QA pairs in a file"""
        try:
            with open(file_path, 'r') as f:
                question_set = json.load(f)
        except Exception as e:
            print(f"Error loading evaluation file: {str(e)}")
            return []

        results = []
        for item in question_set:
            result = self.evaluate_single(
                question=item["question"],
                reference_answer=item["reference_answer"],
                verbose=verbose
            )
            if result:
                results.append(result)
        
        if results:
            self._print_summary_statistics(results)
        
        return results

    def _print_summary_statistics(self, results: List[Dict]):
        """Print aggregated evaluation metrics"""
        bert_f1_scores = [r['metrics']['bertscore']['f1'] for r in results]
        rouge_l_f1 = [r['metrics']['rouge']['rougeL']['f1'] for r in results]
        
        print("\n" + "=" * 80)
        print("Evaluation Summary")
        print("=" * 80)
        print(f"Number of questions evaluated: {len(results)}")
        print(f"Average BERTScore F1: {sum(bert_f1_scores)/len(bert_f1_scores):.3f}")
        print(f"Average ROUGE-L F1: {sum(rouge_l_f1)/len(rouge_l_f1):.3f}")
        print("=" * 80)
