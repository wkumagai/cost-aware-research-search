import os
import json
import time
import random
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def create_logical_reasoning_problems() -> List[Dict]:
    """Create a set of logical reasoning problems with known correct answers."""
    problems = [
        {
            "problem": "All birds can fly. Penguins are birds. Can penguins fly?",
            "correct_answer": "No",
            "explanation": "The premise is false - not all birds can fly, penguins cannot fly"
        },
        {
            "problem": "If it rains, the ground gets wet. The ground is wet. Did it rain?",
            "correct_answer": "Maybe",
            "explanation": "Wet ground could have other causes besides rain"
        },
        {
            "problem": "All cats are animals. Some animals are dogs. Are some cats dogs?",
            "correct_answer": "No",
            "explanation": "Cats and dogs are separate categories"
        },
        {
            "problem": "If A > B and B > C, then A > C. A = 5, B = 3, C = 1. Is A > C?",
            "correct_answer": "Yes",
            "explanation": "5 > 3 > 1, so 5 > 1 is true"
        },
        {
            "problem": "All squares are rectangles. Some rectangles are not squares. Are all rectangles squares?",
            "correct_answer": "No",
            "explanation": "Some rectangles are not squares, so not all rectangles are squares"
        },
        {
            "problem": "If today is Monday, then tomorrow is Tuesday. Today is not Monday. What day is tomorrow?",
            "correct_answer": "Unknown",
            "explanation": "We only know what happens if today is Monday, but it's not Monday"
        },
        {
            "problem": "Every student in the class passed the test. John is in the class. Did John pass the test?",
            "correct_answer": "Yes",
            "explanation": "If every student passed and John is a student, then John passed"
        },
        {
            "problem": "Some flowers are red. Some red things are cars. Are some flowers cars?",
            "correct_answer": "No",
            "explanation": "No logical connection between flowers and cars"
        },
        {
            "problem": "If it's sunny, people go to the beach. People are at the beach. Is it sunny?",
            "correct_answer": "Maybe",
            "explanation": "People could be at the beach for reasons other than sunshine"
        },
        {
            "problem": "All metals conduct electricity. Gold is a metal. Does gold conduct electricity?",
            "correct_answer": "Yes",
            "explanation": "If all metals conduct electricity and gold is a metal, then gold conducts electricity"
        },
        {
            "problem": "No reptiles are mammals. Some animals are reptiles. Are some animals mammals?",
            "correct_answer": "Maybe",
            "explanation": "The statements don't tell us about non-reptile animals"
        },
        {
            "problem": "If X = 2Y and Y = 3, what is X?",
            "correct_answer": "6",
            "explanation": "X = 2 × 3 = 6"
        },
        {
            "problem": "All roses are flowers. Some flowers smell good. Do all roses smell good?",
            "correct_answer": "No",
            "explanation": "We only know some flowers smell good, not all"
        },
        {
            "problem": "Either it will rain or it will be sunny. It's not raining. What's the weather?",
            "correct_answer": "Sunny",
            "explanation": "If only two options exist and one is eliminated, the other must be true"
        },
        {
            "problem": "All teachers are smart. Mary is smart. Is Mary a teacher?",
            "correct_answer": "Maybe",
            "explanation": "Smart people can exist who aren't teachers"
        }
    ]
    return problems

def create_direct_prompt(problem: str) -> str:
    """Create a direct question-answering prompt."""
    return f"Answer this logical reasoning question: {problem}\n\nProvide only your final answer."

def create_cot_prompt(problem: str) -> str:
    """Create a chain-of-thought reasoning prompt."""
    return f"""Answer this logical reasoning question: {problem}

Please think through this step-by-step:
1. Identify the key premises or given information
2. Determine what logical rules or relationships apply
3. Work through the reasoning process
4. State your conclusion

Provide your step-by-step reasoning, then give your final answer."""

def query_openai(prompt: str, max_retries: int = 3) -> str:
    """Query OpenAI API with error handling and retries."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=150,
                temperature=0.1
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"API call attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                return "ERROR: API call failed"
    return "ERROR: All retries failed"

def extract_answer(response: str) -> str:
    """Extract the final answer from the model response."""
    response = response.lower().strip()
    
    # Look for common answer patterns
    if "yes" in response and "no" not in response:
        return "yes"
    elif "no" in response and "yes" not in response:
        return "no"
    elif "maybe" in response or "unknown" in response or "cannot" in response:
        return "maybe"
    elif any(char.isdigit() for char in response):
        # Extract numbers for numerical answers
        numbers = ''.join(char for char in response if char.isdigit() or char == '.')
        if numbers:
            return numbers
    elif "sunny" in response:
        return "sunny"
    
    # If no clear pattern, return the first few words
    words = response.split()[:3]
    return " ".join(words)

def evaluate_accuracy(predicted: str, correct: str) -> bool:
    """Check if the predicted answer matches the correct answer."""
    predicted = predicted.lower().strip()
    correct = correct.lower().strip()
    
    # Direct match
    if predicted == correct:
        return True
    
    # Flexible matching for common variations
    if correct == "yes" and any(word in predicted for word in ["yes", "true", "correct"]):
        return True
    elif correct == "no" and any(word in predicted for word in ["no", "false", "incorrect", "not"]):
        return True
    elif correct == "maybe" and any(word in predicted for word in ["maybe", "unknown", "possible", "cannot", "unsure"]):
        return True
    elif correct.isdigit() and correct in predicted:
        return True
    elif correct == "sunny" and "sunny" in predicted:
        return True
    
    return False

def run_experiment() -> Dict:
    """Run the complete chain-of-thought experiment."""
    print("Starting Chain-of-Thought Prompting Experiment")
    print("=" * 50)
    
    # Create problems and select subset
    all_problems = create_logical_reasoning_problems()
    selected_problems = random.sample(all_problems, min(15, len(all_problems)))  # Use 15 problems for 30 total API calls
    
    results = {
        "direct_prompting": [],
        "cot_prompting": [],
        "problems": selected_problems
    }
    
    print(f"Testing {len(selected_problems)} problems with both prompt types...")
    
    # Test each problem with both prompt types
    for i, problem in enumerate(selected_problems):
        print(f"\nProblem {i+1}/{len(selected_problems)}")
        
        # Direct prompting
        direct_prompt = create_direct_prompt(problem["problem"])
        direct_response = query_openai(direct_prompt)
        direct_answer = extract_answer(direct_response)
        direct_correct = evaluate_accuracy(direct_answer, problem["correct_answer"])
        
        results["direct_prompting"].append({
            "problem_id": i,
            "response": direct_response,
            "extracted_answer": direct_answer,
            "correct_answer": problem["correct_answer"],
            "is_correct": direct_correct
        })
        
        # Chain-of-thought prompting
        cot_prompt = create_cot_prompt(problem["problem"])
        cot_response = query_openai(cot_prompt)
        cot_answer = extract_answer(cot_response)
        cot_correct = evaluate_accuracy(cot_answer, problem["correct_answer"])
        
        results["cot_prompting"].append({
            "problem_id": i,
            "response": cot_response,
            "extracted_answer": cot_answer,
            "correct_answer": problem["correct_answer"],
            "is_correct": cot_correct
        })
        
        print(f"  Direct: {direct_answer} ({'✓' if direct_correct else '✗'})")
        print(f"  CoT: {cot_answer} ({'✓' if cot_correct else '✗'})")
        
        # Small delay to avoid rate limits
        time.sleep(0.5)
    
    return results

def analyze_results(results: Dict) -> Dict:
    """Analyze the experimental results."""
    direct_accuracy = np.mean([r["is_correct"] for r in results["direct_prompting"]])
    cot_accuracy = np.mean([r["is_correct"] for r in results["cot_prompting"]])
    
    improvement = cot_accuracy - direct_accuracy
    improvement_percentage = improvement * 100
    
    analysis = {
        "direct_accuracy": direct_accuracy,
        "cot_accuracy": cot_accuracy,
        "improvement": improvement,
        "improvement_percentage": improvement_percentage,
        "meets_threshold": improvement_percentage >= 5.0,
        "total_problems": len(results["problems"]),
        "direct_correct_count": sum(r["is_correct"] for r in results["direct_prompting"]),
        "cot_correct_count": sum(r["is_correct"] for r in results["cot_prompting"])
    }
    
    return analysis

def main():
    """Main experiment execution."""
    start_time = time.time()
    
    try:
        # Run experiment
        results = run_experiment()
        
        # Analyze results
        analysis = analyze_results(results)
        
        # Prepare final results
        final_results = {
            "experiment_type": "chain_of_thought_prompting",
            "timestamp": time.time(),
            "duration_seconds": time.time() - start_time,
            "analysis": analysis,
            "detailed_results": results
        }
        
        # Save to JSON
        output_path = "/Users/kumacmini/cost-aware-research-search/results/iter_03_results.json"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        # Print summary table
        print("\n" + "=" * 60)
        print("EXPERIMENT RESULTS SUMMARY")
        print("=" * 60)
        
        summary_data = {
            "Metric": ["Direct Prompting Accuracy", "Chain-of-Thought Accuracy", "Improvement", "Success Threshold Met", "Total Problems Tested"],
            "Value": [
                f"{analysis['direct_accuracy']:.1%} ({analysis['direct_correct_count']}/{analysis['total_problems']})",
                f"{analysis['cot_accuracy']:.1%} ({analysis['cot_correct_count']}/{analysis['total_problems']})",
                f"{analysis['improvement_percentage']:+.1f}%",
                "Yes" if analysis['meets_threshold'] else "No",
                str(analysis['total_problems'])
            ]
        }
        
        df = pd.DataFrame(summary_data)
        print(df.to_string(index=False, max_colwidth=50))
        
        print(f"\nExperiment completed in {time.time() - start_time:.1f} seconds")
        print(f"Results saved to: {output_path}")
        
        # Print conclusion
        if analysis['meets_threshold']:
            print(f"\n✅ HYPOTHESIS SUPPORTED: Chain-of-thought prompting improved accuracy by {analysis['improvement_percentage']:.1f}%")
        else:
            print(f"\n❌ HYPOTHESIS NOT SUPPORTED: Chain-of-thought prompting only improved accuracy by {analysis['improvement_percentage']:.1f}%")
    
    except Exception as e:
        print(f"Experiment failed: {e}")
        # Save error results
        error_results = {
            "experiment_type": "chain_of_thought_prompting",
            "timestamp": time.time(),
            "duration_seconds": time.time() - start_time,
            "error": str(e),
            "status": "failed"
        }
        
        output_path = "/Users/kumacmini/cost-aware-research-search/results/iter_03_results.json"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(error_results, f, indent=2)

if __name__ == "__main__":
    main()