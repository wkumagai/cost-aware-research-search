import os
import json
import time
import re
import numpy as np
from openai import OpenAI
import matplotlib.pyplot as plt
from datetime import datetime

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Test questions covering different task types
TEST_QUESTIONS = [
    "Explain the water cycle in 3 steps.",
    "List 5 benefits of exercise.",
    "Summarize the causes of World War I.",
    "What are the main components of a computer?",
    "Describe how photosynthesis works.",
    "Compare democracy and autocracy.",
    "What factors affect climate change?",
    "Explain the scientific method."
]

# Prompt conditions
PROMPT_CONDITIONS = {
    "unformatted": "Answer the following question: {question}",
    "numbered_list": "Answer the following question using a numbered list format:\n{question}\n\nFormat your response as:\n1. [point]\n2. [point]\n3. [point]",
    "bullet_points": "Answer the following question using bullet points:\n{question}\n\nFormat your response as:\n• [point]\n• [point]\n• [point]",
    "structured_template": "Answer the following question using this structured format:\n{question}\n\n**Main Answer:**\n[your main response]\n\n**Key Points:**\n1. [point 1]\n2. [point 2]\n3. [point 3]"
}

def get_llm_response(prompt, max_retries=3):
    """Get response from OpenAI with retry logic."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=200,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Failed to get response after {max_retries} attempts: {e}")
                return None
            time.sleep(1)

def calculate_formatting_compliance(response, condition_type):
    """Calculate compliance score based on formatting requirements."""
    if response is None:
        return 0.0
    
    response = response.strip()
    
    if condition_type == "unformatted":
        # For unformatted, just check if we got a reasonable response
        return 1.0 if len(response) > 10 else 0.0
    
    elif condition_type == "numbered_list":
        # Check for numbered list pattern
        numbered_pattern = re.findall(r'^\d+\.\s+', response, re.MULTILINE)
        has_numbers = len(numbered_pattern) >= 2
        has_structure = bool(re.search(r'\d+\.\s+.*\n.*\d+\.\s+', response))
        return (0.6 if has_numbers else 0.0) + (0.4 if has_structure else 0.0)
    
    elif condition_type == "bullet_points":
        # Check for bullet point pattern
        bullet_pattern = re.findall(r'^[•\-\*]\s+', response, re.MULTILINE)
        has_bullets = len(bullet_pattern) >= 2
        has_structure = bool(re.search(r'[•\-\*]\s+.*\n.*[•\-\*]\s+', response))
        return (0.6 if has_bullets else 0.0) + (0.4 if has_structure else 0.0)
    
    elif condition_type == "structured_template":
        # Check for structured template elements
        has_main_answer = "**Main Answer:**" in response or "Main Answer:" in response
        has_key_points = "**Key Points:**" in response or "Key Points:" in response
        has_numbered_points = bool(re.search(r'\d+\.\s+', response))
        
        score = 0.0
        if has_main_answer: score += 0.4
        if has_key_points: score += 0.3
        if has_numbered_points: score += 0.3
        return score
    
    return 0.0

def run_experiment():
    """Run the complete formatting experiment."""
    print("Starting LLM Response Formatting Experiment")
    print("=" * 50)
    
    results = {
        "experiment_info": {
            "timestamp": datetime.now().isoformat(),
            "total_questions": len(TEST_QUESTIONS),
            "conditions": list(PROMPT_CONDITIONS.keys()),
            "model": "gpt-3.5-turbo"
        },
        "condition_results": {},
        "summary_stats": {}
    }
    
    total_calls = 0
    
    # Run experiment for each condition
    for condition, prompt_template in PROMPT_CONDITIONS.items():
        print(f"\nTesting condition: {condition}")
        condition_scores = []
        condition_responses = []
        
        for i, question in enumerate(TEST_QUESTIONS):
            if total_calls >= 30:  # API limit check
                print("Reached API call limit, stopping early")
                break
                
            prompt = prompt_template.format(question=question)
            
            print(f"  Question {i+1}: {question[:50]}...")
            
            response = get_llm_response(prompt)
            total_calls += 1
            
            if response:
                compliance_score = calculate_formatting_compliance(response, condition)
                condition_scores.append(compliance_score)
                condition_responses.append({
                    "question": question,
                    "response": response,
                    "compliance_score": compliance_score
                })
                print(f"    Compliance Score: {compliance_score:.2f}")
            else:
                condition_scores.append(0.0)
                condition_responses.append({
                    "question": question,
                    "response": None,
                    "compliance_score": 0.0
                })
                print("    Failed to get response")
            
            # Small delay to avoid rate limits
            time.sleep(0.1)
        
        # Store results for this condition
        results["condition_results"][condition] = {
            "responses": condition_responses,
            "mean_score": np.mean(condition_scores) if condition_scores else 0.0,
            "std_score": np.std(condition_scores) if condition_scores else 0.0,
            "n_samples": len(condition_scores)
        }
        
        print(f"  Condition '{condition}' - Mean Score: {np.mean(condition_scores):.3f}")
    
    # Calculate summary statistics
    baseline_score = results["condition_results"]["unformatted"]["mean_score"]
    
    summary_stats = {}
    for condition in PROMPT_CONDITIONS.keys():
        if condition in results["condition_results"]:
            mean_score = results["condition_results"][condition]["mean_score"]
            improvement = ((mean_score - baseline_score) / baseline_score * 100) if baseline_score > 0 else 0
            
            summary_stats[condition] = {
                "mean_score": mean_score,
                "improvement_vs_baseline": improvement,
                "meets_threshold": improvement >= 20.0 if condition != "unformatted" else True
            }
    
    results["summary_stats"] = summary_stats
    results["total_api_calls"] = total_calls
    
    return results

def print_results_table(results):
    """Print a formatted results table."""
    print("\n" + "=" * 70)
    print("EXPERIMENT RESULTS SUMMARY")
    print("=" * 70)
    
    print(f"Total API Calls: {results['total_api_calls']}")
    print(f"Model: {results['experiment_info']['model']}")
    print()
    
    # Results table
    print("Condition".ljust(20) + "Mean Score".ljust(12) + "Improvement".ljust(12) + "Threshold Met")
    print("-" * 60)
    
    baseline_score = results["summary_stats"]["unformatted"]["mean_score"]
    
    for condition, stats in results["summary_stats"].items():
        mean_score = f"{stats['mean_score']:.3f}"
        
        if condition == "unformatted":
            improvement = "baseline"
            threshold_met = "N/A"
        else:
            improvement = f"{stats['improvement_vs_baseline']:+.1f}%"
            threshold_met = "✓" if stats["meets_threshold"] else "✗"
        
        print(condition.ljust(20) + mean_score.ljust(12) + improvement.ljust(12) + threshold_met)
    
    print("\n" + "=" * 70)
    
    # Success analysis
    successful_conditions = [
        cond for cond, stats in results["summary_stats"].items() 
        if cond != "unformatted" and stats["meets_threshold"]
    ]
    
    if successful_conditions:
        print(f"SUCCESS: {len(successful_conditions)} condition(s) met the 20% improvement threshold:")
        for cond in successful_conditions:
            improvement = results["summary_stats"][cond]["improvement_vs_baseline"]
            print(f"  • {cond}: {improvement:+.1f}% improvement")
    else:
        print("No conditions met the 20% improvement threshold.")
    
    print("\nHypothesis Status:", "SUPPORTED" if successful_conditions else "NOT SUPPORTED")

def main():
    try:
        # Run the experiment
        results = run_experiment()
        
        # Print results table
        print_results_table(results)
        
        # Save results to JSON file
        output_path = "/Users/kumacmini/cost-aware-research-search/results/iter_02_results.json"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")
        
    except Exception as e:
        print(f"Experiment failed with error: {e}")
        # Save error info
        error_results = {
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
            "status": "failed"
        }
        
        try:
            output_path = "/Users/kumacmini/cost-aware-research-search/results/iter_02_results.json"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(error_results, f, indent=2)
        except:
            pass

if __name__ == "__main__":
    main()