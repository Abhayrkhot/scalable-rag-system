#!/usr/bin/env python3
"""
Comprehensive evaluation script for the RAG system
"""
import asyncio
import os
import sys
import json
from pathlib import Path

# Add the app directory to Python path
sys.path.append(str(Path(__file__).parent))

from app.evaluation.ragas_evaluator import EvaluationHarness
from app.services.advanced_query_service import AdvancedQueryService
from app.services.massive_ingestion_service import MassiveIngestionService
import structlog

# Configure logging
structlog.configure(
    processors=[
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ],
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

class RAGEvaluator:
    def __init__(self):
        self.evaluation_harness = EvaluationHarness()
        self.query_service = AdvancedQueryService()
        self.ingestion_service = MassiveIngestionService()
    
    async def run_comprehensive_evaluation(self):
        """Run comprehensive evaluation suite"""
        print("🔍 Starting Comprehensive RAG Evaluation")
        print("=" * 50)
        
        # Test data for evaluation
        test_data = await self._prepare_test_data()
        
        # Run evaluation
        results = await self.evaluation_harness.run_comprehensive_evaluation(test_data)
        
        # Display results
        self._display_results(results)
        
        # Save results
        await self._save_results(results)
        
        return results
    
    async def _prepare_test_data(self):
        """Prepare test data for evaluation"""
        # Sample questions and expected answers
        questions = [
            "What is machine learning?",
            "How do neural networks work?",
            "What are the different types of machine learning?",
            "Explain deep learning concepts",
            "What is the difference between supervised and unsupervised learning?"
        ]
        
        ground_truths = [
            "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed.",
            "Neural networks are computing systems inspired by biological neural networks that process information through interconnected nodes.",
            "The main types of machine learning are supervised learning, unsupervised learning, and reinforcement learning.",
            "Deep learning is a subset of machine learning that uses neural networks with multiple layers to model and understand complex patterns.",
            "Supervised learning uses labeled training data, while unsupervised learning finds patterns in data without labels."
        ]
        
        # Generate answers using the RAG system
        answers = []
        contexts = []
        
        for question in questions:
            try:
                result = await self.query_service.answer_question_advanced(
                    question=question,
                    collection_name="test_collection",
                    top_k=5,
                    use_reranking=True,
                    use_query_expansion=True,
                    use_hybrid_search=True
                )
                
                answers.append(result["answer"])
                contexts.append([source["content"] for source in result["sources"]])
                
            except Exception as e:
                logger.error(f"Error generating answer for '{question}': {e}")
                answers.append("Error generating answer")
                contexts.append([])
        
        return {
            "questions": questions,
            "ground_truths": ground_truths,
            "answers": answers,
            "contexts": contexts
        }
    
    def _display_results(self, results):
        """Display evaluation results"""
        print("\n📊 Evaluation Results")
        print("=" * 30)
        
        if "ragas" in results:
            ragas_results = results["ragas"]
            print(f"Overall Score: {ragas_results.get('overall_score', 0):.3f}")
            print("\nRAGAS Metrics:")
            for metric, score in ragas_results.get('metrics', {}).items():
                print(f"  {metric}: {score:.3f}")
            
            print("\nInterpretations:")
            for metric, interpretation in ragas_results.get('interpretation', {}).items():
                print(f"  {metric}: {interpretation}")
            
            print("\nRecommendations:")
            for rec in ragas_results.get('recommendations', []):
                print(f"  - {rec}")
        
        if "retrieval" in results:
            retrieval_results = results["retrieval"]
            print(f"\nRetrieval Metrics:")
            for metric, score in retrieval_results.items():
                print(f"  {metric}: {score:.3f}")
        
        if "overall" in results:
            overall = results["overall"]
            print(f"\nOverall Performance: {overall.get('performance_level', 'unknown')}")
            print(f"Overall Score: {overall.get('overall_score', 0):.3f}")
    
    async def _save_results(self, results):
        """Save evaluation results to file"""
        results_file = Path("evaluation_results.json")
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to {results_file}")

async def main():
    """Main evaluation function"""
    evaluator = RAGEvaluator()
    
    try:
        results = await evaluator.run_comprehensive_evaluation()
        print("\n✅ Evaluation completed successfully!")
        
        # Check if results meet minimum thresholds
        if "ragas" in results:
            overall_score = results["ragas"].get("overall_score", 0)
            if overall_score >= 0.7:
                print("🎉 System performance is excellent!")
            elif overall_score >= 0.5:
                print("👍 System performance is good")
            else:
                print("⚠️  System performance needs improvement")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        print(f"❌ Evaluation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Set environment variables for testing
    os.environ.setdefault("API_KEY", "test-key")
    os.environ.setdefault("OPENAI_API_KEY", "your-openai-key-here")
    os.environ.setdefault("VECTOR_DB_PROVIDER", "chroma")
    os.environ.setdefault("CHROMA_PERSIST_DIRECTORY", "./test_data/chroma_db")
    
    print("⚠️  Please set your OPENAI_API_KEY environment variable before running!")
    print("   export OPENAI_API_KEY='your-actual-openai-key'")
    print()
    
    asyncio.run(main())
