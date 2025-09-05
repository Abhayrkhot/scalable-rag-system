#!/usr/bin/env python3
"""
Comprehensive evaluation script for the RAG system
"""
import asyncio
import os
import sys
import json
import argparse
from pathlib import Path

# Add the app directory to Python path
sys.path.append(str(Path(__file__).parent))

from app.evaluation.ragas_evaluator import EvaluationHarness
from app.services.query_service import QueryService
from app.services.ingestion_service import IngestionService
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
        self.query_service = QueryService()
        self.ingestion_service = IngestionService()
    
    async def run_comprehensive_evaluation(self, dataset_path: str = None):
        """Run comprehensive evaluation suite"""
        print("🔍 Starting Comprehensive RAG Evaluation")
        print("=" * 50)
        
        # Load test data
        if dataset_path and os.path.exists(dataset_path):
            test_data = await self._load_dataset(dataset_path)
        else:
            test_data = await self._prepare_test_data()
        
        # Run evaluation
        results = await self.evaluation_harness.run_comprehensive_evaluation(test_data)
        
        # Display results
        self._display_results(results)
        
        # Save results
        await self._save_results(results)
        
        return results
    
    async def _load_dataset(self, dataset_path: str) -> Dict[str, Any]:
        """Load evaluation dataset from file"""
        try:
            with open(dataset_path, 'r') as f:
                if dataset_path.endswith('.json'):
                    return json.load(f)
                elif dataset_path.endswith('.jsonl'):
                    data = []
                    for line in f:
                        data.append(json.loads(line.strip()))
                    return self._convert_jsonl_to_test_data(data)
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            return await self._prepare_test_data()
    
    def _convert_jsonl_to_test_data(self, data: List[Dict]) -> Dict[str, Any]:
        """Convert JSONL data to test data format"""
        questions = []
        ground_truths = []
        answers = []
        contexts = []
        
        for item in data:
            questions.append(item.get('question', ''))
            ground_truths.append(item.get('ground_truth', item.get('answer', '')))
            answers.append(item.get('answer', ''))
            contexts.append(item.get('context', []))
        
        return {
            "questions": questions,
            "ground_truths": ground_truths,
            "answers": answers,
            "contexts": contexts
        }
    
    async def _prepare_test_data(self) -> Dict[str, Any]:
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
                result = await self.query_service.answer_question(
                    question=question,
                    collection_name="test_collection",
                    top_k=5,
                    use_reranking=True,
                    use_query_expansion=True,
                    use_hybrid=True
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
        
        # Also save markdown report
        self._save_markdown_report(results)
    
    def _save_markdown_report(self, results):
        """Save markdown report"""
        report_file = Path("reports/evaluation_report.md")
        report_file.parent.mkdir(exist_ok=True)
        
        with open(report_file, 'w') as f:
            f.write("# RAG System Evaluation Report\n\n")
            f.write(f"Generated on: {asyncio.get_event_loop().time()}\n\n")
            
            if "ragas" in results:
                ragas = results["ragas"]
                f.write("## RAGAS Metrics\n\n")
                f.write(f"**Overall Score:** {ragas.get('overall_score', 0):.3f}\n\n")
                
                f.write("### Detailed Metrics\n\n")
                for metric, score in ragas.get('metrics', {}).items():
                    f.write(f"- **{metric}:** {score:.3f}\n")
                
                f.write("\n### Recommendations\n\n")
                for rec in ragas.get('recommendations', []):
                    f.write(f"- {rec}\n")
            
            if "retrieval" in results:
                retrieval = results["retrieval"]
                f.write("\n## Retrieval Metrics\n\n")
                for metric, score in retrieval.items():
                    f.write(f"- **{metric}:** {score:.3f}\n")
        
        print(f"📄 Markdown report saved to {report_file}")

async def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description="Run RAG system evaluation")
    parser.add_argument("--dataset", help="Path to evaluation dataset (JSON/JSONL)")
    parser.add_argument("--collection", default="test_collection", help="Collection name to test")
    
    args = parser.parse_args()
    
    evaluator = RAGEvaluator()
    
    try:
        results = await evaluator.run_comprehensive_evaluation(args.dataset)
        print("\n✅ Evaluation completed successfully!")
        
        # Check if results meet minimum thresholds
        if "ragas" in results:
            overall_score = results["ragas"].get("overall_score", 0)
            if overall_score >= 0.7:
                print("🎉 System performance is excellent!")
                sys.exit(0)
            elif overall_score >= 0.5:
                print("👍 System performance is good")
                sys.exit(0)
            else:
                print("⚠️  System performance needs improvement")
                sys.exit(1)
        else:
            print("❌ No evaluation results generated")
            sys.exit(1)
        
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
