import json
import os
import time
import random
import statistics
from collections import defaultdict
import openai
import numpy as np

def setup_openai():
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    openai.api_key = api_key

def generate_boolean_expression(difficulty='medium'):
    """Generate Boolean algebra problems with explicit variable definitions"""
    operators = ['AND', 'OR', 'NOT', 'XOR']
    variables = ['A', 'B', 'C', 'D']
    
    if difficulty == 'easy':
        num_vars = 2
        num_ops = 1
    elif difficulty == 'medium':
        num_vars = 3
        num_ops = 2
    else:  # hard
        num_vars = 3
        num_ops = 3
    
    # Select variables and assign values
    selected_vars = random.sample(variables, num_vars)
    var_values = {var: random.choice([True, False]) for var in selected_vars}
    
    # Generate expression
    if num_ops == 1:
        if random.choice([True, False]) and len(selected_vars) >= 2:
            expr = f"{selected_vars[0]} {random.choice(['AND', 'OR'])} {selected_vars[1]}"
        else:
            expr = f"NOT {selected_vars[0]}"
    else:
        # Build more complex expressions
        expr_parts = [selected_vars[0]]
        remaining_vars = selected_vars[1:]
        
        for i in range(min(num_ops, len(remaining_vars))):
            op = random.choice(['AND', 'OR'])
            if random.random() < 0.3:  # 30% chance of NOT
                expr_parts.append(f"{op} NOT {remaining_vars[i]}")
            else:
                expr_parts.append(f"{op} {remaining_vars[i]}")
        
        expr = " ".join(expr_parts)
    
    # Calculate correct answer
    eval_expr = expr.replace('AND', 'and').replace('OR', 'or').replace('NOT', 'not').replace('XOR', '^')
    for var, val in var_values.items():
        eval_expr = eval_expr.replace(var, str(val))
    
    try:
        correct_answer = eval(eval_expr)
    except:
        # Fallback for XOR
        if 'XOR' in expr:
            # Simple XOR case
            if 'A XOR B' in expr:
                correct_answer = var_values['A'] != var_values['B']
            else:
                correct_answer = True  # Default fallback
        else:
            correct_answer = True
    
    return {
        'expression': expr,
        'variables': var_values,
        'correct_answer': correct_answer,
        'difficulty': difficulty
    }

def create_direct_prompt(problem):
    """Create direct answer prompt"""
    var_text = ", ".join([f"{k}={v}" for k, v in problem['variables'].items()])
    return f"Given {var_text}, evaluate: {problem['expression']}\n\nAnswer (True or False):"

def create_structured_prompt(problem):
    """Create structured step-by-step prompt"""
    var_text = ", ".join([f"{k}={v}" for k, v in problem['variables'].items()])
    return f"""Given {var_text}, evaluate: {problem['expression']}

Please solve this step by step:
1. First, identify the variables and their values
2. Then, work through the expression from left to right (respecting operator precedence)
3. Show your work for each operation
4. State the final result

Final Answer (True or False):"""

def create_explicit_prompt(problem):
    """Create explicit reasoning prompt with detailed instructions"""
    var_text = ", ".join([f"{k}={v}" for k, v in problem['variables'].items()])
    return f"""Boolean Logic Problem:
Expression: {problem['expression']}
Variable Values: {var_text}

Instructions:
- Substitute each variable with its given value
- Apply Boolean operations in correct order (NOT first, then AND/OR left to right)
- Show each substitution and calculation step
- Verify your final answer

Step-by-step solution:"""

def call_openai_api(prompt, max_retries=3):
    """Call OpenAI API with retry logic"""
    for attempt in range(max_retries):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=200,
                temperature=0.1
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"API call failed (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return None

def extract_boolean_answer(response):
    """Extract True/False from response"""
    if not response:
        return None
    
    response_lower = response.lower()
    
    # Look for explicit True/False
    if 'true' in response_lower and 'false' not in response_lower:
        return True
    elif 'false' in response_lower and 'true' not in response_lower:
        return False
    elif response_lower.strip() in ['1', 'yes']:
        return True
    elif response_lower.strip() in ['0', 'no']:
        return False
    
    # If both or neither, return None for ambiguous
    return None

def evaluate_response_quality(response, correct_answer):
    """Evaluate response quality and correctness"""
    if not response:
        return {'correct': False, 'confidence': 0, 'explanation_length': 0}
    
    extracted_answer = extract_boolean_answer(response)
    is_correct = extracted_answer == correct_answer if extracted_answer is not None else False
    
    # Measure explanation quality
    explanation_length = len(response.split())
    
    # Simple confidence based on clarity
    confidence = 1.0 if extracted_answer is not None else 0.5
    if 'step' in response.lower() or 'because' in response.lower():
        confidence += 0.2
    
    return {
        'correct': is_correct,
        'confidence': float(confidence),
        'explanation_length': explanation_length,
        'extracted_answer': extracted_answer
    }

def run_experiment():
    """Run the complete experiment"""
    print("Starting Boolean Logic Reasoning Experiment...")
    setup_openai()
    
    # Generate test problems
    problems = []
    difficulties = ['easy', 'medium', 'hard']
    
    # Generate balanced set
    for difficulty in difficulties:
        for _ in range(20):  # 60 total problems
            problems.append(generate_boolean_expression(difficulty))
    
    random.shuffle(problems)
    problems = problems[:60]  # Ensure exactly 60
    
    conditions = ['direct', 'structured', 'explicit']
    results = []
    api_calls = 0
    max_api_calls = 30
    
    print(f"Generated {len(problems)} problems")
    print("Running experiment with 3 conditions...")
    
    # Test each problem with each condition
    for i, problem in enumerate(problems):
        if api_calls >= max_api_calls:
            print(f"Reached max API calls limit ({max_api_calls})")
            break
            
        print(f"Processing problem {i+1}/{len(problems)}...")
        
        for condition in conditions:
            if api_calls >= max_api_calls:
                break
                
            # Create appropriate prompt
            if condition == 'direct':
                prompt = create_direct_prompt(problem)
            elif condition == 'structured':
                prompt = create_structured_prompt(problem)
            else:  # explicit
                prompt = create_explicit_prompt(problem)
            
            # Get response
            response = call_openai_api(prompt)
            api_calls += 1
            
            if response:
                quality = evaluate_response_quality(response, problem['correct_answer'])
                
                results.append({
                    'problem_id': i,
                    'condition': condition,
                    'difficulty': problem['difficulty'],
                    'expression': problem['expression'],
                    'variables': problem['variables'],
                    'correct_answer': problem['correct_answer'],
                    'response': response,
                    'extracted_answer': quality['extracted_answer'],
                    'correct': quality['correct'],
                    'confidence': quality['confidence'],
                    'explanation_length': quality['explanation_length'],
                    'api_calls_used': api_calls
                })
            
            time.sleep(0.1)  # Rate limiting
    
    print(f"Completed experiment with {len(results)} data points using {api_calls} API calls")
    
    if len(results) < 50:
        print(f"WARNING: Only collected {len(results)} data points, minimum required is 50")
    
    return results

def analyze_results(results):
    """Analyze experimental results"""
    if not results:
        return {}
    
    # Group by condition
    by_condition = defaultdict(list)
    for result in results:
        by_condition[result['condition']].append(result)
    
    analysis = {}
    
    for condition, data in by_condition.items():
        if not data:
            continue
            
        accuracies = [r['correct'] for r in data if 'correct' in r]
        confidences = [r['confidence'] for r in data if 'confidence' in r]
        lengths = [r['explanation_length'] for r in data if 'explanation_length' in r]
        
        analysis[condition] = {
            'n_samples': len(data),
            'accuracy_mean': float(np.mean(accuracies)) if accuracies else 0.0,
            'accuracy_std': float(np.std(accuracies)) if len(accuracies) > 1 else 0.0,
            'confidence_mean': float(np.mean(confidences)) if confidences else 0.0,
            'confidence_std': float(np.std(confidences)) if len(confidences) > 1 else 0.0,
            'explanation_length_mean': float(np.mean(lengths)) if lengths else 0.0,
            'explanation_length_std': float(np.std(lengths)) if len(lengths) > 1 else 0.0
        }
    
    return analysis

def print_results_table(analysis):
    """Print formatted results table"""
    print("\n" + "="*80)
    print("EXPERIMENTAL RESULTS")
    print("="*80)
    
    if not analysis:
        print("No results to display")
        return
    
    print(f"{'Condition':<12} {'N':<4} {'Accuracy':<12} {'Acc_Std':<10} {'Confidence':<12} {'Conf_Std':<10} {'Exp_Length':<12}")
    print("-" * 80)
    
    for condition, stats in analysis.items():
        print(f"{condition:<12} {stats['n_samples']:<4} {stats['accuracy_mean']:.3f}      {stats['accuracy_std']:.3f}     {stats['confidence_mean']:.3f}      {stats['confidence_std']:.3f}     {stats['explanation_length_mean']:.1f}")
    
    # Calculate variance reduction
    if 'direct' in analysis and 'structured' in analysis:
        direct_var = analysis['direct']['accuracy_std'] ** 2
        structured_var = analysis['structured']['accuracy_std'] ** 2
        if direct_var > 0:
            variance_reduction = (direct_var - structured_var) / direct_var * 100
            print(f"\nVariance Reduction (structured vs direct): {variance_reduction:.1f}%")
    
    print("="*80)

def main():
    start_time = time.time()
    
    try:
        # Run experiment
        results = run_experiment()
        
        # Analyze results
        analysis = analyze_results(results)
        
        # Print results
        print_results_table(analysis)
        
        # Prepare data for JSON serialization
        json_data = {
            'experiment_info': {
                'total_samples': len(results),
                'conditions': ['direct', 'structured', 'explicit'],
                'timestamp': time.time(),
                'duration_seconds': float(time.time() - start_time)
            },
            'raw_results': results,
            'analysis': analysis,
            'hypothesis_test': {
                'metric': 'response_consistency_score',
                'baseline': 'direct_answer_prompts',
                'success_threshold': '20% reduction in response variance'
            }
        }
        
        # Save results
        output_path = '/Users/kumacmini/cost-aware-research-search/results/iter_04_results.json'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(json_data, f, indent=2, default=str)
        
        print(f"\nResults saved to: {output_path}")
        print(f"Total runtime: {time.time() - start_time:.1f} seconds")
        
    except Exception as e:
        print(f"Experiment failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()