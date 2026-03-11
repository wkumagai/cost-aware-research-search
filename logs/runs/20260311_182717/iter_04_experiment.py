import os
import json
import time
import numpy as np
import pandas as pd
from openai import OpenAI
import random
from typing import List, Dict, Tuple
import re

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def generate_constraint_tasks() -> List[Dict]:
    """Generate a variety of constraint tasks for testing."""
    tasks = []
    
    # Word limit tasks
    word_limits = [10, 25, 50]
    for limit in word_limits:
        tasks.append({
            'type': 'word_limit',
            'task': f'Explain photosynthesis in {limit} words.',
            'constraint': limit,
            'validation_func': lambda text, lim=limit: len(text.split()) <= lim
        })
    
    # Number range tasks
    number_ranges = [(1, 10), (50, 100), (200, 300)]
    for min_val, max_val in number_ranges:
        tasks.append({
            'type': 'number_range',
            'task': f'Pick a random number between {min_val} and {max_val}.',
            'constraint': (min_val, max_val),
            'validation_func': lambda text, min_v=min_val, max_v=max_val: validate_number_in_range(text, min_v, max_v)
        })
    
    # List length tasks
    list_lengths = [3, 5, 7]
    for length in list_lengths:
        tasks.append({
            'type': 'list_length',
            'task': f'List {length} benefits of exercise.',
            'constraint': length,
            'validation_func': lambda text, len_val=length: validate_list_length(text, len_val)
        })
    
    # Format tasks (bullet points)
    tasks.append({
        'type': 'bullet_format',
        'task': 'List 4 programming languages using bullet points.',
        'constraint': 'bullet_points',
        'validation_func': validate_bullet_format
    })
    
    return tasks

def validate_number_in_range(text: str, min_val: int, max_val: int) -> bool:
    """Check if text contains a number within the specified range."""
    numbers = re.findall(r'\b\d+\b', text)
    if not numbers:
        return False
    try:
        num = int(numbers[0])
        return min_val <= num <= max_val
    except ValueError:
        return False

def validate_list_length(text: str, expected_length: int) -> bool:
    """Check if text contains a list of the expected length."""
    # Count numbered items, bullet points, or line breaks
    numbered_items = len(re.findall(r'^\d+\.', text, re.MULTILINE))
    bullet_items = len(re.findall(r'^[•\-\*]', text, re.MULTILINE))
    line_items = len([line for line in text.split('\n') if line.strip() and not line.strip().endswith(':')])
    
    return max(numbered_items, bullet_items, line_items) == expected_length

def validate_bullet_format(text: str) -> bool:
    """Check if text uses proper bullet point formatting."""
    bullet_patterns = [r'^[•\-\*]', r'^\s*[•\-\*]']
    for pattern in bullet_patterns:
        if len(re.findall(pattern, text, re.MULTILINE)) >= 3:
            return True
    return False

def create_prompt_variants(task: str) -> Dict[str, str]:
    """Create different prompt variants for the same task."""
    return {
        'baseline': task,
        'exactly': task.replace('in ', 'in exactly ').replace('List ', 'List exactly '),
        'only': f"Please provide only what is requested. {task}",
        'precisely': task.replace('in ', 'in precisely ').replace('List ', 'List precisely ')
    }

def query_llm(prompt: str, max_retries: int = 2) -> str:
    """Query OpenAI API with error handling."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=150,
                temperature=0.3
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"API error (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                return ""
    return ""

def run_experiment() -> Dict:
    """Run the complete constraint adherence experiment."""
    print("Starting constraint adherence experiment...")
    
    # Generate tasks (reduced for API limit)
    all_tasks = generate_constraint_tasks()
    selected_tasks = random.sample(all_tasks, min(10, len(all_tasks)))  # Limit to 10 tasks
    
    results = {
        'experiment_config': {
            'n_tasks': len(selected_tasks),
            'n_conditions': 4,
            'total_queries': len(selected_tasks) * 4
        },
        'raw_results': [],
        'summary_stats': {}
    }
    
    condition_results = {condition: [] for condition in ['baseline', 'exactly', 'only', 'precisely']}
    
    print(f"Testing {len(selected_tasks)} tasks across 4 conditions...")
    
    for i, task in enumerate(selected_tasks):
        print(f"Processing task {i+1}/{len(selected_tasks)}: {task['type']}")
        
        prompt_variants = create_prompt_variants(task['task'])
        
        for condition, prompt in prompt_variants.items():
            print(f"  Testing condition: {condition}")
            
            response = query_llm(prompt)
            if not response:
                print(f"    Failed to get response for {condition}")
                adherence = False
            else:
                try:
                    adherence = task['validation_func'](response)
                except Exception as e:
                    print(f"    Validation error: {e}")
                    adherence = False
            
            result_entry = {
                'task_id': i,
                'task_type': task['type'],
                'condition': condition,
                'prompt': prompt,
                'response': response,
                'constraint_met': adherence,
                'constraint_value': str(task['constraint'])
            }
            
            results['raw_results'].append(result_entry)
            condition_results[condition].append(adherence)
            
            time.sleep(0.5)  # Rate limiting
    
    # Calculate summary statistics
    for condition in condition_results:
        adherence_rate = np.mean(condition_results[condition]) if condition_results[condition] else 0
        results['summary_stats'][condition] = {
            'adherence_rate': adherence_rate,
            'n_samples': len(condition_results[condition]),
            'n_successful': sum(condition_results[condition])
        }
    
    return results

def print_results_table(results: Dict):
    """Print a formatted results table."""
    print("\n" + "="*60)
    print("CONSTRAINT ADHERENCE EXPERIMENT RESULTS")
    print("="*60)
    
    # Create summary DataFrame
    summary_data = []
    for condition, stats in results['summary_stats'].items():
        summary_data.append({
            'Condition': condition.title(),
            'Adherence Rate': f"{stats['adherence_rate']:.1%}",
            'Success Count': f"{stats['n_successful']}/{stats['n_samples']}",
            'Sample Size': stats['n_samples']
        })
    
    df = pd.DataFrame(summary_data)
    print(df.to_string(index=False))
    
    # Calculate improvements over baseline
    baseline_rate = results['summary_stats']['baseline']['adherence_rate']
    print(f"\nBaseline adherence rate: {baseline_rate:.1%}")
    print("\nImprovement over baseline:")
    
    for condition in ['exactly', 'only', 'precisely']:
        condition_rate = results['summary_stats'][condition]['adherence_rate']
        improvement = ((condition_rate - baseline_rate) / baseline_rate * 100) if baseline_rate > 0 else 0
        print(f"  {condition.title()}: {improvement:+.1f}%")
    
    # Check success threshold
    best_improvement = max([
        ((results['summary_stats'][cond]['adherence_rate'] - baseline_rate) / baseline_rate * 100) 
        if baseline_rate > 0 else 0
        for cond in ['exactly', 'only', 'precisely']
    ])
    
    print(f"\nBest improvement: {best_improvement:.1f}%")
    print(f"Success threshold (15% improvement): {'✓ MET' if best_improvement >= 15 else '✗ NOT MET'}")
    
    # Task type breakdown
    print(f"\nTask breakdown:")
    task_types = {}
    for result in results['raw_results']:
        task_type = result['task_type']
        if task_type not in task_types:
            task_types[task_type] = {'total': 0, 'success': 0}
        task_types[task_type]['total'] += 1
        if result['constraint_met']:
            task_types[task_type]['success'] += 1
    
    for task_type, counts in task_types.items():
        success_rate = counts['success'] / counts['total'] if counts['total'] > 0 else 0
        print(f"  {task_type}: {success_rate:.1%} ({counts['success']}/{counts['total']})")

def main():
    """Main execution function."""
    try:
        start_time = time.time()
        
        # Run experiment
        results = run_experiment()
        
        # Print results
        print_results_table(results)
        
        # Add metadata
        results['metadata'] = {
            'experiment_duration_seconds': time.time() - start_time,
            'total_api_calls': len(results['raw_results']),
            'hypothesis': "Adding specific constraint keywords improves LLM adherence to constraints",
            'success_threshold_met': any([
                ((results['summary_stats'][cond]['adherence_rate'] - results['summary_stats']['baseline']['adherence_rate']) 
                 / results['summary_stats']['baseline']['adherence_rate'] * 100) >= 15
                if results['summary_stats']['baseline']['adherence_rate'] > 0 else False
                for cond in ['exactly', 'only', 'precisely']
            ])
        }
        
        # Save results
        output_path = '/Users/kumacmini/cost-aware-research-search/results/iter_04_results.json'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")
        print(f"Experiment completed in {time.time() - start_time:.1f} seconds")
        
    except Exception as e:
        print(f"Experiment failed with error: {e}")
        # Save error info
        error_results = {
            'error': str(e),
            'status': 'failed',
            'partial_results': locals().get('results', {})
        }
        
        output_path = '/Users/kumacmini/cost-aware-research-search/results/iter_04_results.json'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(error_results, f, indent=2)

if __name__ == "__main__":
    main()