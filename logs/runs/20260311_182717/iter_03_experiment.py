import os
import json
import time
import random
import numpy as np
import pandas as pd
from openai import OpenAI
from typing import List, Dict, Tuple
import re

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Math word problems dataset
MATH_PROBLEMS = [
    {
        "problem": "A bakery sold 234 cupcakes on Monday and 187 cupcakes on Tuesday. If each cupcake costs $3.50, how much money did they make in total?",
        "answer": 1473.50,
        "category": "multiplication_addition"
    },
    {
        "problem": "Sarah has 45 stickers. She gives away 1/3 of them to her friends and buys 12 more. How many stickers does she have now?",
        "answer": 42,
        "category": "fractions"
    },
    {
        "problem": "A rectangular garden is 8 meters long and 6 meters wide. If fencing costs $15 per meter, how much will it cost to fence the entire perimeter?",
        "answer": 420,
        "category": "geometry"
    },
    {
        "problem": "Tom drives 120 km in 2 hours, then 180 km in 3 hours. What is his average speed for the entire trip?",
        "answer": 60,
        "category": "averages"
    },
    {
        "problem": "A store offers a 25% discount on a jacket that originally costs $80. After the discount, a 8% sales tax is added. What is the final price?",
        "answer": 64.80,
        "category": "percentages"
    },
    {
        "problem": "Lisa has twice as many apples as oranges. If she has 24 apples, and she buys 6 more oranges, how many fruits does she have in total?",
        "answer": 42,
        "category": "algebra"
    },
    {
        "problem": "A train leaves at 9:30 AM and travels for 4 hours 45 minutes. What time does it arrive?",
        "answer": "2:15 PM",
        "category": "time"
    },
    {
        "problem": "In a class of 30 students, 18 like pizza, 12 like burgers, and 5 like both. How many students like neither pizza nor burgers?",
        "answer": 5,
        "category": "sets"
    },
    {
        "problem": "A recipe calls for 2.5 cups of flour to make 20 cookies. How many cups of flour are needed to make 48 cookies?",
        "answer": 6,
        "category": "proportions"
    },
    {
        "problem": "Mark saves $25 per week. After 8 weeks, he spends $120 on a gift. How much money does he have left?",
        "answer": 80,
        "category": "basic_arithmetic"
    },
    {
        "problem": "A box contains 15 red balls and 10 blue balls. If you pick 2 balls without replacement, what is the probability both are red?",
        "answer": 0.42,
        "category": "probability"
    },
    {
        "problem": "Water flows into a tank at 12 liters per minute and flows out at 8 liters per minute. How long will it take to fill a 120-liter empty tank?",
        "answer": 30,
        "category": "rates"
    },
    {
        "problem": "A company's profit increased from $50,000 to $65,000. What is the percentage increase?",
        "answer": 30,
        "category": "percentages"
    },
    {
        "problem": "Three friends split a restaurant bill of $84.60 equally. If they leave a 18% tip, how much does each person pay in total?",
        "answer": 33.21,
        "category": "division_percentages"
    },
    {
        "problem": "A circular pool has a radius of 5 meters. If it costs $8 per square meter to cover it, what is the total cost? (Use π = 3.14)",
        "answer": 628,
        "category": "geometry"
    }
]

def create_direct_prompt(problem: str) -> str:
    return f"Solve this math problem and provide only the final numerical answer:\n\n{problem}"

def create_chain_of_thought_prompt(problem: str) -> str:
    return f"""Solve this math problem step by step. Show your reasoning clearly, then provide the final numerical answer.

Problem: {problem}

Please follow this format:
1. Identify what we need to find
2. List the given information
3. Show each calculation step
4. State the final answer clearly"""

def call_openai(prompt: str, max_retries: int = 3) -> str:
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=200,
                temperature=0.1
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"API call failed after {max_retries} attempts: {e}")
                return "ERROR"
            time.sleep(2 ** attempt)
    return "ERROR"

def extract_numerical_answer(response: str) -> float:
    try:
        # Look for patterns like "answer is X", "= X", "X dollars", etc.
        patterns = [
            r'(?:answer|result|total|final answer|solution)(?:\s+is)?:?\s*\$?(\d+(?:\.\d+)?)',
            r'(?:^|\s)(\d+(?:\.\d+)?)\s*(?:dollars?|meters?|minutes?|hours?|%)?(?:\s|$)',
            r'=\s*\$?(\d+(?:\.\d+)?)',
            r'\$(\d+(?:\.\d+)?)',
            r'(\d+(?:\.\d+)?)\s*(?:is the|answer)',
            r'(\d+:\d+\s*(?:AM|PM))'  # For time answers
        ]
        
        # Special handling for time format
        time_match = re.search(r'(\d+:\d+\s*(?:AM|PM))', response, re.IGNORECASE)
        if time_match:
            return time_match.group(1)
            
        # Extract all numbers and take the last significant one
        numbers = re.findall(r'\d+(?:\.\d+)?', response)
        if numbers:
            return float(numbers[-1])
            
        return None
    except:
        return None

def calculate_accuracy(predicted, actual) -> bool:
    if predicted is None:
        return False
    
    # Handle string answers (like time)
    if isinstance(actual, str):
        return str(predicted).strip().upper() == actual.strip().upper()
    
    # Handle numerical answers with tolerance
    try:
        predicted_num = float(predicted)
        actual_num = float(actual)
        return abs(predicted_num - actual_num) < 0.01 * abs(actual_num) + 0.01
    except:
        return False

def score_reasoning_coherence(response: str) -> int:
    """Score reasoning coherence from 0-5"""
    score = 0
    
    # Check for step-by-step structure
    if any(marker in response.lower() for marker in ['step', '1.', '2.', 'first', 'then', 'next']):
        score += 1
    
    # Check for problem identification
    if any(phrase in response.lower() for phrase in ['find', 'calculate', 'determine', 'need to']):
        score += 1
    
    # Check for given information identification
    if any(phrase in response.lower() for phrase in ['given', 'we know', 'information']):
        score += 1
    
    # Check for mathematical operations
    if any(op in response for op in ['+', '-', '*', '/', '=', '×', '÷']):
        score += 1
    
    # Check for clear final answer
    if any(phrase in response.lower() for phrase in ['final answer', 'answer is', 'therefore', 'result']):
        score += 1
    
    return score

def run_experiment() -> Dict:
    print("Starting Chain-of-Thought vs Direct Answer experiment...")
    print(f"Testing on {len(MATH_PROBLEMS)} math problems")
    
    # Select subset of problems for the experiment
    selected_problems = random.sample(MATH_PROBLEMS, min(15, len(MATH_PROBLEMS)))
    
    results = {
        "direct_answers": [],
        "chain_of_thought_answers": [],
        "problems": selected_problems,
        "timestamp": time.time()
    }
    
    total_calls = 0
    
    for i, problem_data in enumerate(selected_problems):
        problem = problem_data["problem"]
        correct_answer = problem_data["answer"]
        
        print(f"Processing problem {i+1}/{len(selected_problems)}")
        
        # Test direct answer approach
        if total_calls < 25:
            direct_prompt = create_direct_prompt(problem)
            direct_response = call_openai(direct_prompt)
            total_calls += 1
            
            direct_extracted = extract_numerical_answer(direct_response)
            direct_accuracy = calculate_accuracy(direct_extracted, correct_answer)
            direct_coherence = score_reasoning_coherence(direct_response)
            
            results["direct_answers"].append({
                "problem_id": i,
                "response": direct_response,
                "extracted_answer": direct_extracted,
                "correct_answer": correct_answer,
                "accuracy": direct_accuracy,
                "coherence_score": direct_coherence
            })
        
        # Test chain-of-thought approach
        if total_calls < 25:
            cot_prompt = create_chain_of_thought_prompt(problem)
            cot_response = call_openai(cot_prompt)
            total_calls += 1
            
            cot_extracted = extract_numerical_answer(cot_response)
            cot_accuracy = calculate_accuracy(cot_extracted, correct_answer)
            cot_coherence = score_reasoning_coherence(cot_response)
            
            results["chain_of_thought_answers"].append({
                "problem_id": i,
                "response": cot_response,
                "extracted_answer": cot_extracted,
                "correct_answer": correct_answer,
                "accuracy": cot_accuracy,
                "coherence_score": cot_coherence
            })
        
        # Add small delay to avoid rate limits
        time.sleep(0.5)
    
    print(f"Total API calls made: {total_calls}")
    return results

def analyze_results(results: Dict) -> Dict:
    analysis = {}
    
    # Direct answer metrics
    direct_accuracies = [r["accuracy"] for r in results["direct_answers"]]
    direct_coherence = [r["coherence_score"] for r in results["direct_answers"]]
    
    # Chain-of-thought metrics
    cot_accuracies = [r["accuracy"] for r in results["chain_of_thought_answers"]]
    cot_coherence = [r["coherence_score"] for r in results["chain_of_thought_answers"]]
    
    analysis["direct_accuracy"] = np.mean(direct_accuracies) * 100
    analysis["direct_coherence"] = np.mean(direct_coherence)
    analysis["cot_accuracy"] = np.mean(cot_accuracies) * 100
    analysis["cot_coherence"] = np.mean(cot_coherence)
    
    analysis["accuracy_improvement"] = analysis["cot_accuracy"] - analysis["direct_accuracy"]
    analysis["coherence_improvement"] = analysis["cot_coherence"] - analysis["direct_coherence"]
    
    analysis["success_threshold_met"] = analysis["accuracy_improvement"] >= 10.0
    
    # Statistical tests (basic)
    analysis["sample_size"] = len(direct_accuracies)
    
    return analysis

def main():
    try:
        # Run the experiment
        results = run_experiment()
        
        # Analyze results
        analysis = analyze_results(results)
        
        # Combine results and analysis
        final_results = {
            "experiment_data": results,
            "analysis": analysis,
            "metadata": {
                "experiment_type": "chain_of_thought_vs_direct",
                "model": "gpt-3.5-turbo",
                "timestamp": time.time(),
                "problems_tested": len(results.get("direct_answers", [])),
                "hypothesis_confirmed": analysis["success_threshold_met"]
            }
        }
        
        # Save to JSON
        output_path = "/Users/kumacmini/cost-aware-research-search/results/iter_03_results.json"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(final_results, f, indent=2, default=str)
        
        # Print summary table
        print("\n" + "="*60)
        print("EXPERIMENT RESULTS SUMMARY")
        print("="*60)
        print(f"Sample Size: {analysis['sample_size']} problems per condition")
        print(f"Model Used: gpt-3.5-turbo")
        print("-" * 60)
        print(f"{'Metric':<30} {'Direct':<15} {'Chain-of-Thought':<15}")
        print("-" * 60)
        print(f"{'Accuracy (%)':<30} {analysis['direct_accuracy']:<15.1f} {analysis['cot_accuracy']:<15.1f}")
        print(f"{'Coherence Score (0-5)':<30} {analysis['direct_coherence']:<15.1f} {analysis['cot_coherence']:<15.1f}")
        print("-" * 60)
        print(f"{'Accuracy Improvement':<30} {analysis['accuracy_improvement']:<15.1f} percentage points")
        print(f"{'Coherence Improvement':<30} {analysis['coherence_improvement']:<15.1f} points")
        print("-" * 60)
        print(f"Success Threshold (10% improvement): {'✓ MET' if analysis['success_threshold_met'] else '✗ NOT MET'}")
        print(f"Hypothesis Confirmed: {'YES' if analysis['success_threshold_met'] else 'NO'}")
        print("="*60)
        print(f"Results saved to: {output_path}")
        
    except Exception as e:
        print(f"Experiment failed with error: {e}")
        # Save error information
        error_results = {
            "error": str(e),
            "timestamp": time.time(),
            "status": "failed"
        }
        
        output_path = "/Users/kumacmini/cost-aware-research-search/results/iter_03_results.json"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(error_results, f, indent=2)

if __name__ == "__main__":
    main()