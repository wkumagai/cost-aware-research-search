import os
import json
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from openai import OpenAI
from typing import List, Dict, Tuple, Any
import re

def generate_logical_problems() -> List[Dict[str, Any]]:
    """Generate synthetic logical reasoning problems of varying complexity."""
    
    # Simple logical problems
    simple_problems = [
        {
            "problem": "If all cats are animals and Fluffy is a cat, what can we conclude about Fluffy?",
            "correct_answer": "Fluffy is an animal",
            "complexity": "simple",
            "reasoning_steps": ["All cats are animals (given)", "Fluffy is a cat (given)", "Therefore, Fluffy is an animal (syllogism)"]
        },
        {
            "problem": "If it's raining, then the ground is wet. It's raining. What can we conclude?",
            "correct_answer": "The ground is wet",
            "complexity": "simple",
            "reasoning_steps": ["If raining → ground wet (given)", "It's raining (given)", "Therefore, ground is wet (modus ponens)"]
        },
        {
            "problem": "All birds can fly. Penguins are birds. Can penguins fly according to this logic?",
            "correct_answer": "Yes, according to the given premises",
            "complexity": "simple",
            "reasoning_steps": ["All birds can fly (given)", "Penguins are birds (given)", "Therefore, penguins can fly (syllogism)"]
        },
        {
            "problem": "If A implies B, and B implies C, and A is true, what can we conclude about C?",
            "correct_answer": "C is true",
            "complexity": "simple",
            "reasoning_steps": ["A → B (given)", "B → C (given)", "A is true (given)", "Therefore C is true (chain rule)"]
        },
        {
            "problem": "Either it's Monday or Tuesday. It's not Monday. What day is it?",
            "correct_answer": "Tuesday",
            "complexity": "simple",
            "reasoning_steps": ["Monday OR Tuesday (given)", "NOT Monday (given)", "Therefore Tuesday (disjunctive syllogism)"]
        }
    ]
    
    # Compound logical problems
    compound_problems = [
        {
            "problem": "If (A and B) implies C, and (C or D) implies E, and both A and B are true, and D is false, what can we conclude about E?",
            "correct_answer": "E is true",
            "complexity": "compound",
            "reasoning_steps": ["(A ∧ B) → C (given)", "(C ∨ D) → E (given)", "A is true, B is true (given)", "D is false (given)", "A ∧ B is true", "Therefore C is true", "C ∨ D is true (since C is true)", "Therefore E is true"]
        },
        {
            "problem": "In a group: All lawyers are smart. Some smart people are wealthy. No wealthy people are lazy. John is a lazy lawyer. Is this scenario logically consistent?",
            "correct_answer": "No, it's inconsistent",
            "complexity": "compound",
            "reasoning_steps": ["All lawyers are smart", "John is a lawyer", "Therefore John is smart", "No wealthy people are lazy", "John is lazy", "Therefore John is not wealthy", "Some smart people are wealthy (doesn't create contradiction)", "But John being a 'lazy lawyer' contradicts 'all lawyers are smart' if we assume lazy people can't be smart"]
        },
        {
            "problem": "If P then (Q and R). If (Q and R) then (S or T). If S then not U. If T then U. P is true and U is false. What can we determine about S and T?",
            "correct_answer": "S is true and T is false",
            "complexity": "compound",
            "reasoning_steps": ["P is true", "P → (Q ∧ R), so Q ∧ R is true", "(Q ∧ R) → (S ∨ T), so S ∨ T is true", "U is false", "T → U, so if T were true, U would be true", "Since U is false, T must be false", "Since S ∨ T is true and T is false, S must be true"]
        },
        {
            "problem": "Every student who studies hard passes. Every student who passes gets a good job. Some students who get good jobs are happy. No happy people are stressed. Maria is a stressed student who studies hard. What can we conclude about Maria's happiness?",
            "correct_answer": "Maria is not happy",
            "complexity": "compound",
            "reasoning_steps": ["Maria studies hard", "Every student who studies hard passes", "So Maria passes", "Every student who passes gets a good job", "So Maria gets a good job", "Maria is stressed", "No happy people are stressed", "Therefore Maria is not happy"]
        },
        {
            "problem": "In a logic puzzle: If the red door leads to treasure, then the blue door is locked. If the blue door is locked, then either the green door is open or the treasure is fake. The green door is not open and the treasure is real. What can we conclude about the red door?",
            "correct_answer": "The red door does not lead to treasure",
            "complexity": "compound",
            "reasoning_steps": ["Red door → Blue door locked", "Blue door locked → (Green door open OR Treasure fake)", "Green door NOT open", "Treasure is real (NOT fake)", "If blue door locked, then green door open OR treasure fake", "Since neither condition is met, blue door is NOT locked", "If blue door NOT locked, then red door does NOT lead to treasure"]
        }
    ]
    
    # Select problems for the experiment
    selected_simple = random.sample(simple_problems, 3)
    selected_compound = random.sample(compound_problems, 3)
    
    all_problems = selected_simple + selected_compound
    random.shuffle(all_problems)
    
    return all_problems

def create_prompts(problem: Dict[str, Any]) -> Dict[str, str]:
    """Create direct and reasoning-step prompts for a problem."""
    
    direct_prompt = f"""Solve this logical reasoning problem:

{problem['problem']}

Provide your answer directly."""

    reasoning_prompt = f"""Solve this logical reasoning problem step by step:

{problem['problem']}

Please work through this systematically:
1. Identify the given premises
2. Apply logical rules step by step
3. State your final conclusion

Show your reasoning clearly before giving your final answer."""

    return {
        "direct": direct_prompt,
        "reasoning": reasoning_prompt
    }

def evaluate_logical_consistency(response: str, correct_answer: str, expected_reasoning: List[str]) -> Dict[str, float]:
    """Evaluate the logical consistency of a response."""
    
    response_lower = response.lower()
    correct_lower = correct_answer.lower()
    
    # Basic correctness score
    correctness_score = 0.0
    if any(word in response_lower for word in correct_lower.split()):
        correctness_score = 0.5
    
    # Check for key concepts
    key_concepts = ["true", "false", "therefore", "because", "since", "implies", "conclude"]
    concept_score = sum(1 for concept in key_concepts if concept in response_lower) / len(key_concepts)
    
    # Check for reasoning structure
    reasoning_indicators = ["first", "second", "given", "premise", "conclusion", "step"]
    structure_score = sum(1 for indicator in reasoning_indicators if indicator in response_lower) / len(reasoning_indicators)
    
    # Overall consistency score
    consistency_score = (correctness_score * 0.5) + (concept_score * 0.3) + (structure_score * 0.2)
    
    return {
        "consistency_score": min(consistency_score, 1.0),
        "correctness_score": correctness_score,
        "concept_score": concept_score,
        "structure_score": structure_score
    }

def get_llm_response(prompt: str, client: OpenAI) -> str:
    """Get response from OpenAI API with error handling."""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=300,
            temperature=0.1
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"API Error: {e}")
        return "Error: Could not generate response"

def run_experiment():
    """Run the complete experiment."""
    
    # Initialize OpenAI client
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    # Generate problems
    problems = generate_logical_problems()
    
    # Initialize results storage
    results = {
        "experiment_config": {
            "hypothesis": "Adding explicit reasoning steps to prompts will improve logical consistency scores",
            "n_problems": len(problems),
            "conditions": ["direct", "reasoning"],
            "complexity_levels": ["simple", "compound"]
        },
        "detailed_results": [],
        "summary_stats": {}
    }
    
    all_scores = []
    api_calls = 0
    max_calls = 25
    
    print("Running logical reasoning experiment...")
    print(f"Total problems: {len(problems)}")
    
    for i, problem in enumerate(problems):
        if api_calls >= max_calls:
            print(f"Reached API limit ({max_calls} calls), stopping early")
            break
            
        print(f"\nProcessing problem {i+1}/{len(problems)} ({problem['complexity']})")
        
        prompts = create_prompts(problem)
        problem_results = {
            "problem_id": i,
            "problem": problem['problem'],
            "complexity": problem['complexity'],
            "correct_answer": problem['correct_answer'],
            "conditions": {}
        }
        
        for condition_name, prompt in prompts.items():
            if api_calls >= max_calls:
                break
                
            print(f"  Testing {condition_name} condition...")
            
            response = get_llm_response(prompt, client)
            api_calls += 1
            
            scores = evaluate_logical_consistency(
                response, 
                problem['correct_answer'],
                problem['reasoning_steps']
            )
            
            problem_results["conditions"][condition_name] = {
                "response": response,
                "scores": scores
            }
            
            all_scores.append({
                "problem_id": i,
                "complexity": problem['complexity'],
                "condition": condition_name,
                **scores
            })
            
            time.sleep(0.5)  # Rate limiting
    
    results["detailed_results"] = all_scores
    
    # Calculate summary statistics
    df = pd.DataFrame(all_scores)
    
    if len(df) > 0:
        summary = {}
        
        # Overall statistics
        for condition in df['condition'].unique():
            condition_data = df[df['condition'] == condition]
            summary[condition] = {
                "mean_consistency": float(condition_data['consistency_score'].mean()),
                "std_consistency": float(condition_data['consistency_score'].std()),
                "n_samples": len(condition_data)
            }
        
        # By complexity
        for complexity in df['complexity'].unique():
            complexity_data = df[df['complexity'] == complexity]
            summary[f"{complexity}_complexity"] = {}
            for condition in complexity_data['condition'].unique():
                subset = complexity_data[complexity_data['condition'] == condition]
                if len(subset) > 0:
                    summary[f"{complexity}_complexity"][condition] = {
                        "mean_consistency": float(subset['consistency_score'].mean()),
                        "n_samples": len(subset)
                    }
        
        results["summary_stats"] = summary
        
        # Calculate improvement
        direct_scores = df[df['condition'] == 'direct']['consistency_score']
        reasoning_scores = df[df['condition'] == 'reasoning']['consistency_score']
        
        if len(direct_scores) > 0 and len(reasoning_scores) > 0:
            improvement = (reasoning_scores.mean() - direct_scores.mean()) / direct_scores.mean() * 100
            results["improvement_percentage"] = float(improvement)
            results["hypothesis_supported"] = improvement > 15.0
    
    # Save results
    output_path = "/Users/kumacmini/cost-aware-research-search/results/iter_01_results.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    return results, df

def print_summary_table(results: Dict, df: pd.DataFrame):
    """Print a clear summary table of results."""
    
    print("\n" + "="*80)
    print("LOGICAL REASONING EXPERIMENT RESULTS")
    print("="*80)
    
    if len(df) == 0:
        print("No results to display - experiment failed")
        return
    
    print(f"\nTotal API calls made: {len(df)}")
    print(f"Problems tested: {len(df['problem_id'].unique())}")
    
    # Overall results table
    print("\nOVERALL RESULTS:")
    print("-" * 50)
    
    summary_data = []
    for condition in ['direct', 'reasoning']:
        condition_data = df[df['condition'] == condition]
        if len(condition_data) > 0:
            summary_data.append({
                'Condition': condition.title(),
                'Mean Consistency Score': f"{condition_data['consistency_score'].mean():.3f}",
                'Std Dev': f"{condition_data['consistency_score'].std():.3f}",
                'N Samples': len(condition_data)
            })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))
    
    # By complexity
    print("\nRESULTS BY COMPLEXITY:")
    print("-" * 50)
    
    for complexity in ['simple', 'compound']:
        complexity_data = df[df['complexity'] == complexity]
        if len(complexity_data) > 0:
            print(f"\n{complexity.upper()} PROBLEMS:")
            comp_summary = []
            for condition in ['direct', 'reasoning']:
                subset = complexity_data[complexity_data['condition'] == condition]
                if len(subset) > 0:
                    comp_summary.append({
                        'Condition': condition.title(),
                        'Mean Score': f"{subset['consistency_score'].mean():.3f}",
                        'N': len(subset)
                    })
            
            if comp_summary:
                comp_df = pd.DataFrame(comp_summary)
                print(comp_df.to_string(index=False))
    
    # Hypothesis test results
    if 'improvement_percentage' in results:
        print(f"\nHYPOTHESIS TEST:")
        print("-" * 50)
        print(f"Improvement: {results['improvement_percentage']:.1f}%")
        print(f"Success threshold: 15%")
        print(f"Hypothesis supported: {'YES' if results.get('hypothesis_supported', False) else 'NO'}")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    try:
        results, df = run_experiment()
        print_summary_table(results, df)
        print(f"\nResults saved to: /Users/kumacmini/cost-aware-research-search/results/iter_01_results.json")
        
    except Exception as e:
        print(f"Experiment failed with error: {e}")
        # Save error info
        error_results = {
            "error": str(e),
            "experiment_status": "failed",
            "timestamp": time.time()
        }
        
        output_path = "/Users/kumacmini/cost-aware-research-search/results/iter_01_results.json"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(error_results, f, indent=2)