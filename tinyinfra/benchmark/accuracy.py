"""
Accuracy benchmark using MMLU dataset
"""
import torch
from typing import Dict, Optional, List
from tqdm import tqdm
import random


class AccuracyBenchmark:
    """Evaluate model accuracy on MMLU"""
    
    def __init__(self, model):
        self.model = model
    
    def run(
        self,
        split: str = 'validation',
        num_samples: Optional[int] = None,
        subjects: Optional[List[str]] = None,
        seed: int = 42
    ) -> Dict:
        """
        Run MMLU evaluation
        
        Args:
            split: 'validation' or 'test'
            num_samples: Limit samples (None = all)
            subjects: List of subjects to test (None = all)
            seed: Random seed for sampling
            
        Returns:
            Results dict with accuracy metrics
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "datasets library required. Install: pip install datasets"
            )
        
        print(f"\n📊 MMLU Accuracy Benchmark")
        print(f"   Split: {split}")
        
        # Load dataset
        print(f"\n⏳ Loading MMLU dataset...")
        
        if subjects:
            # Load specific subjects
            all_data = []
            for subject in subjects:
                dataset = load_dataset("cais/mmlu", subject, split=split)
                all_data.extend(list(dataset))
            print(f"   Subjects: {len(subjects)}")
        else:
            # Load all subjects
            dataset = load_dataset("cais/mmlu", "all", split=split)
            all_data = list(dataset)
            print(f"   Subjects: all (57)")
        
        # Sample if needed
        if num_samples and num_samples < len(all_data):
            random.seed(seed)
            all_data = random.sample(all_data, num_samples)
            print(f"   Sampled: {num_samples} questions")
        
        print(f"   Total questions: {len(all_data)}")
        
        # Evaluate
        print(f"\n⚡ Evaluating...")
        results = self._evaluate(all_data)
        
        return results
    
    def _evaluate(self, dataset: List[Dict]) -> Dict:
        """Evaluate on dataset"""
        correct = 0
        total = 0
        
        # Track by subject
        subject_results = {}
        
        for item in tqdm(dataset, desc="Evaluating"):
            question = item['question']
            choices = item['choices']
            answer = item['answer']
            subject = item.get('subject', 'unknown')
            
            # Format prompt
            prompt = self._format_prompt(question, choices)
            
            # Generate answer
            try:
                response = self.model.generate(prompt, max_new_tokens=10)
                pred = self._extract_answer(response, choices)
                
                is_correct = (pred == answer)
                if is_correct:
                    correct += 1
                total += 1
                
                # Track by subject
                if subject not in subject_results:
                    subject_results[subject] = {'correct': 0, 'total': 0}
                subject_results[subject]['total'] += 1
                if is_correct:
                    subject_results[subject]['correct'] += 1
                
            except Exception as e:
                print(f"\n⚠️  Error on question: {e}")
                total += 1
                if subject not in subject_results:
                    subject_results[subject] = {'correct': 0, 'total': 0}
                subject_results[subject]['total'] += 1
        
        # Compute accuracies
        overall_accuracy = (correct / total * 100) if total > 0 else 0.0
        
        subject_accuracies = {}
        for subject, stats in subject_results.items():
            acc = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0.0
            subject_accuracies[subject] = {
                'accuracy': acc,
                'correct': stats['correct'],
                'total': stats['total']
            }
        
        return {
            'overall_accuracy': overall_accuracy,
            'correct': correct,
            'total': total,
            'subject_accuracies': subject_accuracies
        }
    
    def _format_prompt(self, question: str, choices: List[str]) -> str:
        """Format MMLU question as prompt"""
        prompt = f"Answer the following multiple choice question.\n\n"
        prompt += f"Question: {question}\n\n"
        prompt += "Choices:\n"
        for i, choice in enumerate(choices):
            prompt += f"{chr(65+i)}. {choice}\n"
        prompt += f"\nAnswer (A, B, C, or D): "
        return prompt
    
    def _extract_answer(self, response: str, choices: List[str]) -> int:
        """Extract answer index from model response"""
        response = response.strip().upper()
        
        # Look for A, B, C, D in response
        for i, letter in enumerate(['A', 'B', 'C', 'D'][:len(choices)]):
            if letter in response[:20]:  # Check first 20 chars
                return i
        
        # Fallback: return 0 (A)
        return 0
    
    def print_results(self, results: Dict):
        """Pretty print results"""
        print(f"\n{'='*60}")
        print(f"📊 MMLU ACCURACY RESULTS")
        print(f"{'='*60}")
        
        print(f"\n🎯 Overall Accuracy: {results['overall_accuracy']:.2f}%")
        print(f"   Correct: {results['correct']}/{results['total']}")
        
        # Subject breakdown (top 10)
        if results.get('subject_accuracies'):
            print(f"\n📚 Subject Breakdown (Top 10):")
            sorted_subjects = sorted(
                results['subject_accuracies'].items(),
                key=lambda x: x[1]['accuracy'],
                reverse=True
            )[:10]
            
            for subject, stats in sorted_subjects:
                print(f"   {subject:<40} {stats['accuracy']:>6.2f}% ({stats['correct']}/{stats['total']})")
        
        print(f"\n{'='*60}\n")