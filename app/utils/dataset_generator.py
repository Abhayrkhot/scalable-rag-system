import os
import json
import random
import asyncio
from pathlib import Path
from typing import List, Dict, Any
import structlog
from faker import Faker
import wikipedia
import requests
from datetime import datetime, timedelta

logger = structlog.get_logger()
fake = Faker()

class LargeDatasetGenerator:
    def __init__(self, output_dir: str = "large_dataset"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for different document types
        (self.output_dir / "pdfs").mkdir(exist_ok=True)
        (self.output_dir / "markdown").mkdir(exist_ok=True)
        (self.output_dir / "text").mkdir(exist_ok=True)
        
        # Topics for generating diverse content
        self.topics = [
            "artificial intelligence", "machine learning", "deep learning", "neural networks",
            "computer science", "data science", "programming", "software engineering",
            "cybersecurity", "cloud computing", "blockchain", "cryptocurrency",
            "quantum computing", "robotics", "automation", "big data",
            "natural language processing", "computer vision", "reinforcement learning",
            "database systems", "distributed systems", "microservices", "devops",
            "web development", "mobile development", "game development", "embedded systems",
            "operating systems", "networking", "algorithms", "data structures",
            "mathematics", "statistics", "physics", "chemistry", "biology",
            "medicine", "pharmacology", "genetics", "neuroscience",
            "psychology", "sociology", "economics", "finance", "business",
            "marketing", "management", "leadership", "entrepreneurship",
            "history", "geography", "politics", "law", "philosophy",
            "literature", "art", "music", "film", "sports", "travel"
        ]
    
    async def generate_million_documents(self, target_count: int = 1000000) -> Dict[str, Any]:
        """Generate 1 million diverse documents"""
        logger.info(f"Starting generation of {target_count:,} documents")
        
        # Distribution: 60% text, 30% markdown, 10% PDF
        text_count = int(target_count * 0.6)
        markdown_count = int(target_count * 0.3)
        pdf_count = int(target_count * 0.1)
        
        results = {
            "total_documents": 0,
            "text_files": 0,
            "markdown_files": 0,
            "pdf_files": 0,
            "processing_time": 0,
            "errors": []
        }
        
        start_time = datetime.now()
        
        try:
            # Generate text files (60%)
            logger.info(f"Generating {text_count:,} text files...")
            text_results = await self._generate_text_files(text_count)
            results["text_files"] = text_results["generated"]
            results["errors"].extend(text_results["errors"])
            
            # Generate markdown files (30%)
            logger.info(f"Generating {markdown_count:,} markdown files...")
            markdown_results = await self._generate_markdown_files(markdown_count)
            results["markdown_files"] = markdown_results["generated"]
            results["errors"].extend(markdown_results["errors"])
            
            # Generate PDF files (10%)
            logger.info(f"Generating {pdf_count:,} PDF files...")
            pdf_results = await self._generate_pdf_files(pdf_count)
            results["pdf_files"] = pdf_results["generated"]
            results["errors"].extend(pdf_results["errors"])
            
            results["total_documents"] = results["text_files"] + results["markdown_files"] + results["pdf_files"]
            results["processing_time"] = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"Generation completed: {results['total_documents']:,} documents in {results['processing_time']:.2f} seconds")
            
            # Create metadata file
            await self._create_metadata_file(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error generating documents: {e}")
            results["errors"].append(str(e))
            return results
    
    async def _generate_text_files(self, count: int) -> Dict[str, Any]:
        """Generate text files with diverse content"""
        generated = 0
        errors = []
        
        for i in range(count):
            try:
                topic = random.choice(self.topics)
                content = await self._generate_text_content(topic, i)
                
                filename = f"document_{i:07d}_{topic.replace(' ', '_')}.txt"
                filepath = self.output_dir / "text" / filename
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                generated += 1
                
                if generated % 10000 == 0:
                    logger.info(f"Generated {generated:,} text files...")
                    
            except Exception as e:
                errors.append(f"Error generating text file {i}: {e}")
        
        return {"generated": generated, "errors": errors}
    
    async def _generate_markdown_files(self, count: int) -> Dict[str, Any]:
        """Generate markdown files with structured content"""
        generated = 0
        errors = []
        
        for i in range(count):
            try:
                topic = random.choice(self.topics)
                content = await self._generate_markdown_content(topic, i)
                
                filename = f"document_{i:07d}_{topic.replace(' ', '_')}.md"
                filepath = self.output_dir / "markdown" / filename
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                generated += 1
                
                if generated % 10000 == 0:
                    logger.info(f"Generated {generated:,} markdown files...")
                    
            except Exception as e:
                errors.append(f"Error generating markdown file {i}: {e}")
        
        return {"generated": generated, "errors": errors}
    
    async def _generate_pdf_files(self, count: int) -> Dict[str, Any]:
        """Generate PDF files (simplified - create text files with .pdf extension for now)"""
        generated = 0
        errors = []
        
        for i in range(count):
            try:
                topic = random.choice(self.topics)
                content = await self._generate_text_content(topic, i)
                
                filename = f"document_{i:07d}_{topic.replace(' ', '_')}.pdf"
                filepath = self.output_dir / "pdfs" / filename
                
                # For now, create as text file (in real implementation, use reportlab or similar)
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                generated += 1
                
                if generated % 1000 == 0:
                    logger.info(f"Generated {generated:,} PDF files...")
                    
            except Exception as e:
                errors.append(f"Error generating PDF file {i}: {e}")
        
        return {"generated": generated, "errors": errors}
    
    async def _generate_text_content(self, topic: str, doc_id: int) -> str:
        """Generate realistic text content about a topic"""
        # Generate multiple sections
        sections = []
        
        # Introduction
        intro = f"This comprehensive document explores various aspects of {topic}. "
        intro += f"Document ID: {doc_id:07d}\n\n"
        intro += f"Generated on: {fake.date_time_between(start_date='-2y', end_date='now')}\n"
        intro += f"Author: {fake.name()}\n"
        intro += f"Organization: {fake.company()}\n\n"
        sections.append(intro)
        
        # Main content sections
        for i in range(random.randint(3, 8)):
            section_title = fake.sentence(nb_words=4).rstrip('.')
            section_content = self._generate_section_content(topic, section_title)
            sections.append(f"## {section_title}\n\n{section_content}\n\n")
        
        # Technical details
        if random.random() < 0.7:  # 70% chance
            tech_section = self._generate_technical_section(topic)
            sections.append(f"## Technical Implementation\n\n{tech_section}\n\n")
        
        # Case studies or examples
        if random.random() < 0.5:  # 50% chance
            case_study = self._generate_case_study(topic)
            sections.append(f"## Case Study\n\n{case_study}\n\n")
        
        # Conclusion
        conclusion = f"In conclusion, {topic} represents a significant area of research and development. "
        conclusion += f"The insights presented in this document (ID: {doc_id:07d}) provide valuable "
        conclusion += f"perspectives on current trends and future directions in the field."
        sections.append(f"## Conclusion\n\n{conclusion}\n\n")
        
        return "".join(sections)
    
    async def _generate_markdown_content(self, topic: str, doc_id: int) -> str:
        """Generate structured markdown content"""
        content = f"# {topic.title()}\n\n"
        content += f"**Document ID:** {doc_id:07d}\n"
        content += f"**Generated:** {fake.date_time_between(start_date='-2y', end_date='now')}\n"
        content += f"**Author:** {fake.name()}\n\n"
        
        # Table of contents
        content += "## Table of Contents\n\n"
        content += "1. [Overview](#overview)\n"
        content += "2. [Key Concepts](#key-concepts)\n"
        content += "3. [Implementation](#implementation)\n"
        content += "4. [Best Practices](#best-practices)\n"
        content += "5. [Future Directions](#future-directions)\n\n"
        
        # Overview section
        content += "## Overview\n\n"
        content += f"{topic.title()} is a rapidly evolving field that encompasses various methodologies and approaches. "
        content += f"This document provides a comprehensive analysis of current trends and developments.\n\n"
        
        # Key concepts
        content += "## Key Concepts\n\n"
        concepts = [fake.sentence(nb_words=6) for _ in range(random.randint(5, 10))]
        for i, concept in enumerate(concepts, 1):
            content += f"{i}. **{concept}** - {fake.paragraph(nb_sentences=2)}\n\n"
        
        # Implementation section
        content += "## Implementation\n\n"
        content += f"Implementing solutions in {topic} requires careful consideration of multiple factors:\n\n"
        content += f"- **Scalability**: {fake.paragraph(nb_sentences=1)}\n"
        content += f"- **Performance**: {fake.paragraph(nb_sentences=1)}\n"
        content += f"- **Maintainability**: {fake.paragraph(nb_sentences=1)}\n\n"
        
        # Code example (if applicable)
        if random.random() < 0.3:
            content += "### Code Example\n\n"
            content += "```python\n"
            content += f"# Example implementation for {topic}\n"
            content += f"def process_{topic.replace(' ', '_')}(data):\n"
            content += f"    \"\"\"Process {topic} data efficiently\"\"\"\n"
            content += f"    result = []\n"
            content += f"    for item in data:\n"
            content += f"        processed = transform(item)\n"
            content += f"        result.append(processed)\n"
            content += f"    return result\n"
            content += "```\n\n"
        
        # Best practices
        content += "## Best Practices\n\n"
        practices = [fake.sentence(nb_words=8) for _ in range(random.randint(4, 7))]
        for practice in practices:
            content += f"- {practice}\n"
        content += "\n"
        
        # Future directions
        content += "## Future Directions\n\n"
        content += f"The future of {topic} looks promising with several emerging trends:\n\n"
        trends = [fake.sentence(nb_words=10) for _ in range(random.randint(3, 6))]
        for trend in trends:
            content += f"- {trend}\n"
        content += "\n"
        
        return content
    
    def _generate_section_content(self, topic: str, section_title: str) -> str:
        """Generate content for a specific section"""
        content = f"This section delves into {section_title.lower()} within the context of {topic}. "
        content += f"{fake.paragraph(nb_sentences=3)}\n\n"
        
        # Add some technical details
        if random.random() < 0.6:
            content += f"From a technical perspective, {section_title.lower()} involves several key components:\n\n"
            for _ in range(random.randint(2, 4)):
                component = fake.word().title()
                content += f"- **{component}**: {fake.paragraph(nb_sentences=1)}\n"
            content += "\n"
        
        return content
    
    def _generate_technical_section(self, topic: str) -> str:
        """Generate technical implementation details"""
        content = f"The technical implementation of {topic} solutions requires careful architecture design. "
        content += f"{fake.paragraph(nb_sentences=2)}\n\n"
        
        content += "### Architecture Components\n\n"
        components = ["Data Layer", "Processing Layer", "API Layer", "Storage Layer", "Monitoring Layer"]
        for component in random.sample(components, random.randint(3, 5)):
            content += f"**{component}**: {fake.paragraph(nb_sentences=1)}\n\n"
        
        return content
    
    def _generate_case_study(self, topic: str) -> str:
        """Generate a case study example"""
        company = fake.company()
        content = f"**Case Study: {company}**\n\n"
        content += f"{company} implemented a {topic} solution to address their specific challenges. "
        content += f"{fake.paragraph(nb_sentences=3)}\n\n"
        
        content += "**Results:**\n"
        results = [
            f"Improved efficiency by {random.randint(20, 80)}%",
            f"Reduced costs by ${random.randint(10000, 100000):,}",
            f"Processed {random.randint(1000, 100000):,} records per day",
            f"Achieved {random.randint(95, 99)}% accuracy"
        ]
        for result in random.sample(results, random.randint(2, 4)):
            content += f"- {result}\n"
        content += "\n"
        
        return content
    
    async def _create_metadata_file(self, results: Dict[str, Any]):
        """Create metadata file with dataset information"""
        metadata = {
            "dataset_info": {
                "total_documents": results["total_documents"],
                "text_files": results["text_files"],
                "markdown_files": results["markdown_files"],
                "pdf_files": results["pdf_files"],
                "generation_time_seconds": results["processing_time"],
                "created_at": datetime.now().isoformat(),
                "topics_covered": self.topics
            },
            "file_structure": {
                "text_directory": "large_dataset/text/",
                "markdown_directory": "large_dataset/markdown/",
                "pdf_directory": "large_dataset/pdfs/"
            },
            "errors": results["errors"]
        }
        
        metadata_file = self.output_dir / "dataset_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Metadata saved to {metadata_file}")

# Usage example
async def main():
    generator = LargeDatasetGenerator()
    results = await generator.generate_million_documents(1000000)  # Generate 1M documents
    print(f"Generated {results['total_documents']:,} documents in {results['processing_time']:.2f} seconds")

if __name__ == "__main__":
    asyncio.run(main())
