import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
import structlog
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevance,
    context_recall,
    context_precision,
    answer_correctness,
    answer_similarity
)
import pandas as pd
import numpy as np

logger = structlog.get_logger()

class RAGASEvaluator:
    def __init__(self):
        self.metrics = [
            faithfulness,
            answer_relevance,
            context_recall,
            context_precision,
            answer_correctness,
            answer_similarity
        ]
    
    async def evaluate_rag_system(self, 
                                questions: List[str],
                                ground_truths: List[str],
                                answers: List[str],
                                contexts: List[List[str]],
                                ground_truth_contexts: List[List[str]] = None) -> Dict[str, Any]:
        """Evaluate RAG system using RAGAS metrics"""
        try:
            # Prepare dataset for RAGAS
            dataset = self._prepare_dataset(
                questions, ground_truths, answers, contexts, ground_truth_contexts
            )
            
            # Run evaluation
            logger.info("Starting RAGAS evaluation...")
            result = await self._run_evaluation(dataset)
            
            # Process results
            evaluation_results = self._process_results(result)
            
            logger.info(f"RAGAS evaluation completed: {evaluation_results['overall_score']:.3f}")
            return evaluation_results
            
        except Exception as e:
            logger.error(f"RAGAS evaluation failed: {e}")
            return {"error": str(e)}
    
    def _prepare_dataset(self, 
                        questions: List[str],
                        ground_truths: List[str],
                        answers: List[str],
                        contexts: List[List[str]],
                        ground_truth_contexts: List[List[str]] = None) -> Dataset:
        """Prepare dataset for RAGAS evaluation"""
        
        # Convert contexts to strings
        context_strings = ["\n".join(ctx) for ctx in contexts]
        
        # Prepare ground truth contexts if not provided
        if ground_truth_contexts is None:
            ground_truth_contexts = contexts
        
        gt_context_strings = ["\n".join(ctx) for ctx in ground_truth_contexts]
        
        # Create dataset
        data = {
            "question": questions,
            "answer": answers,
            "contexts": context_strings,
            "ground_truth": ground_truths,
            "ground_truth_contexts": gt_context_strings
        }
        
        return Dataset.from_dict(data)
    
    async def _run_evaluation(self, dataset: Dataset) -> Dict[str, float]:
        """Run RAGAS evaluation"""
        try:
            # Run evaluation in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                evaluate,
                dataset,
                self.metrics
            )
            
            return result
        except Exception as e:
            logger.error(f"Error running RAGAS evaluation: {e}")
            raise
    
    def _process_results(self, result: Dict[str, float]) -> Dict[str, Any]:
        """Process RAGAS evaluation results"""
        metrics_scores = {}
        
        for metric_name, score in result.items():
            if isinstance(score, (int, float)):
                metrics_scores[metric_name] = float(score)
            else:
                metrics_scores[metric_name] = 0.0
        
        # Calculate overall score
        overall_score = np.mean(list(metrics_scores.values()))
        
        return {
            "overall_score": overall_score,
            "metrics": metrics_scores,
            "interpretation": self._interpret_scores(metrics_scores),
            "recommendations": self._generate_recommendations(metrics_scores)
        }
    
    def _interpret_scores(self, scores: Dict[str, float]) -> Dict[str, str]:
        """Interpret metric scores"""
        interpretations = {}
        
        score_ranges = {
            "excellent": (0.8, 1.0),
            "good": (0.6, 0.8),
            "fair": (0.4, 0.6),
            "poor": (0.0, 0.4)
        }
        
        for metric, score in scores.items():
            for level, (min_score, max_score) in score_ranges.items():
                if min_score <= score < max_score:
                    interpretations[metric] = f"{level} ({score:.3f})"
                    break
        
        return interpretations
    
    def _generate_recommendations(self, scores: Dict[str, float]) -> List[str]:
        """Generate recommendations based on scores"""
        recommendations = []
        
        if scores.get("faithfulness", 0) < 0.7:
            recommendations.append("Improve answer faithfulness by better grounding in context")
        
        if scores.get("answer_relevance", 0) < 0.7:
            recommendations.append("Improve answer relevance by better query understanding")
        
        if scores.get("context_recall", 0) < 0.7:
            recommendations.append("Improve context recall by better retrieval strategies")
        
        if scores.get("context_precision", 0) < 0.7:
            recommendations.append("Improve context precision by better filtering")
        
        if scores.get("answer_correctness", 0) < 0.7:
            recommendations.append("Improve answer correctness by better fact checking")
        
        if not recommendations:
            recommendations.append("System performance is good across all metrics")
        
        return recommendations

class OfflineEvaluator:
    def __init__(self):
        self.evaluator = RAGASEvaluator()
    
    async def evaluate_retrieval(self, 
                               queries: List[str],
                               retrieved_docs: List[List[Dict]],
                               relevant_docs: List[List[Dict]]) -> Dict[str, float]:
        """Evaluate retrieval performance with R@k and MRR"""
        try:
            # Calculate Recall@k for different k values
            k_values = [1, 3, 5, 10]
            recall_scores = {}
            
            for k in k_values:
                recall_scores[f"recall_at_{k}"] = self._calculate_recall_at_k(
                    retrieved_docs, relevant_docs, k
                )
            
            # Calculate MRR
            mrr = self._calculate_mrr(retrieved_docs, relevant_docs)
            
            return {
                **recall_scores,
                "mrr": mrr
            }
            
        except Exception as e:
            logger.error(f"Retrieval evaluation failed: {e}")
            return {}
    
    def _calculate_recall_at_k(self, 
                              retrieved_docs: List[List[Dict]], 
                              relevant_docs: List[List[Dict]], 
                              k: int) -> float:
        """Calculate Recall@k"""
        recalls = []
        
        for ret_docs, rel_docs in zip(retrieved_docs, relevant_docs):
            if not rel_docs:
                continue
            
            # Get top-k retrieved documents
            top_k_docs = ret_docs[:k]
            
            # Calculate recall
            relevant_ids = {doc.get('id', doc.get('source', '')) for doc in rel_docs}
            retrieved_ids = {doc.get('id', doc.get('source', '')) for doc in top_k_docs}
            
            if relevant_ids:
                recall = len(relevant_ids.intersection(retrieved_ids)) / len(relevant_ids)
                recalls.append(recall)
        
        return np.mean(recalls) if recalls else 0.0
    
    def _calculate_mrr(self, 
                      retrieved_docs: List[List[Dict]], 
                      relevant_docs: List[List[Dict]]) -> float:
        """Calculate Mean Reciprocal Rank"""
        reciprocal_ranks = []
        
        for ret_docs, rel_docs in zip(retrieved_docs, relevant_docs):
            if not rel_docs:
                continue
            
            relevant_ids = {doc.get('id', doc.get('source', '')) for doc in rel_docs}
            
            for rank, doc in enumerate(ret_docs, 1):
                doc_id = doc.get('id', doc.get('source', ''))
                if doc_id in relevant_ids:
                    reciprocal_ranks.append(1.0 / rank)
                    break
            else:
                reciprocal_ranks.append(0.0)
        
        return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0

class EvaluationHarness:
    def __init__(self):
        self.ragas_evaluator = RAGASEvaluator()
        self.offline_evaluator = OfflineEvaluator()
    
    async def run_comprehensive_evaluation(self, 
                                         test_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run comprehensive evaluation suite"""
        try:
            results = {}
            
            # RAGAS evaluation
            if "questions" in test_data and "answers" in test_data:
                ragas_results = await self.ragas_evaluator.evaluate_rag_system(
                    questions=test_data["questions"],
                    ground_truths=test_data.get("ground_truths", test_data["questions"]),
                    answers=test_data["answers"],
                    contexts=test_data.get("contexts", [[]] * len(test_data["questions"])),
                    ground_truth_contexts=test_data.get("ground_truth_contexts")
                )
                results["ragas"] = ragas_results
            
            # Retrieval evaluation
            if "retrieved_docs" in test_data and "relevant_docs" in test_data:
                retrieval_results = await self.offline_evaluator.evaluate_retrieval(
                    queries=test_data.get("queries", []),
                    retrieved_docs=test_data["retrieved_docs"],
                    relevant_docs=test_data["relevant_docs"]
                )
                results["retrieval"] = retrieval_results
            
            # Overall assessment
            results["overall"] = self._assess_overall_performance(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Comprehensive evaluation failed: {e}")
            return {"error": str(e)}
    
    def _assess_overall_performance(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall system performance"""
        overall_score = 0.0
        total_metrics = 0
        
        # Aggregate scores
        if "ragas" in results and "overall_score" in results["ragas"]:
            overall_score += results["ragas"]["overall_score"]
            total_metrics += 1
        
        if "retrieval" in results and "mrr" in results["retrieval"]:
            overall_score += results["retrieval"]["mrr"]
            total_metrics += 1
        
        if total_metrics > 0:
            overall_score /= total_metrics
        
        # Determine performance level
        if overall_score >= 0.8:
            performance_level = "excellent"
        elif overall_score >= 0.6:
            performance_level = "good"
        elif overall_score >= 0.4:
            performance_level = "fair"
        else:
            performance_level = "poor"
        
        return {
            "overall_score": overall_score,
            "performance_level": performance_level,
            "total_metrics_evaluated": total_metrics
        }
