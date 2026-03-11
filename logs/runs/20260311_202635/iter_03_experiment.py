import json
import os
import sys
import time
import random
import collections
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import openai

def setup_client():
    """Initialize OpenAI client with API key from environment."""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    return openai.OpenAI(api_key=api_key)

def generate_logical_problem(complexity_level, operators, chain_length):
    """Generate a multi-step logical reasoning problem with known solution."""
    variables = ['P', 'Q', 'R', 'S', 'T']
    
    # Generate truth assignments
    truth_values = {var: random.choice([True, False]) for var in variables[:3]}
    
    # Build logical chain with specific operators and positions
    steps = []
    current_vars = list(truth_values.keys())
    
    for step_idx in range(chain_length):
        if step_idx < len(operators):
            op = operators[step_idx]
            if op == 'NOT':
                var = random.choice(current_vars)
                expr = f"NOT {var}"
                result = not truth_values[var]
            elif op == 'AND':
                var1, var2 = random.sample(current_vars, 2)
                expr = f"{var1} AND {var2}"
                result = truth_values[var1] and truth_values[var2]
            elif op == 'OR':
                var1, var2 = random.sample(current_vars, 2)
                expr = f"{var1} OR {var2}"
                result = truth_values[var1] or truth_values[var2]
            
            # Create new variable for result
            new_var = variables[len(truth_values)]
            truth_values[new_var] = result
            current_vars.append(new_var)
            
            steps.append({
                'step': step_idx + 1,
                'expression': expr,
                'operator': op,
                'position': step_idx,
                'result': result
            })
    
    # Create problem statement
    premise = ", ".join([f"{var} = {truth_values[var]}" for var in variables[:3]])
    problem = f"Given: {premise}\n"
    for i, step in enumerate(steps):
        problem += f"Step {step['step']}: Calculate {step['expression']}\n"
    problem += f"What is the final result of Step {len(steps)}?"
    
    return {
        'problem': problem,
        'truth_values': truth_values,
        'steps': steps,
        'correct_answer': steps[-1]['result'] if steps else None,
        'complexity': complexity_level,
        'chain_length': chain_length,
        'operators_used': operators
    }

def call_llm_with_retry(client, problem_text, max_retries=3):
    """Call LLM with retry logic and error handling."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a logic expert. Solve the multi-step logical reasoning problem step by step. Show your work for each step and provide the final answer as True or False."},
                    {"role": "user", "content": problem_text}
                ],
                max_completion_tokens=200,
                temperature=0.1
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"API call attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return None
    return None

def analyze_llm_response(response_text, correct_steps):
    """Analyze LLM response to identify reasoning errors at specific positions and operators."""
    if not response_text:
        return {'parsing_failed': True, 'errors': []}
    
    errors = []
    
    # Simple pattern matching for step analysis
    lines = response_text.lower().split('\n')
    
    for step_info in correct_steps:
        step_num = step_info['step']
        operator = step_info['operator']
        position = step_info['position']
        correct_result = step_info['result']
        
        # Look for this step in the response
        step_found = False
        for line in lines:
            if f"step {step_num}" in line or f"{step_num}:" in line:
                step_found = True
                # Check if the result appears correct in the line
                if correct_result:
                    if 'true' not in line and 'yes' not in line:
                        errors.append({
                            'step': step_num,
                            'operator': operator,
                            'position': position,
                            'error_type': 'incorrect_result',
                            'expected': correct_result
                        })
                else:
                    if 'false' not in line and 'no' not in line:
                        errors.append({
                            'step': step_num,
                            'operator': operator,
                            'position': position,
                            'error_type': 'incorrect_result',
                            'expected': correct_result
                        })
                break
        
        if not step_found:
            errors.append({
                'step': step_num,
                'operator': operator,
                'position': position,
                'error_type': 'missing_step',
                'expected': correct_result
            })
    
    return {'parsing_failed': False, 'errors': errors}

def run_experiment():
    """Run the main experiment with controlled conditions."""
    print("Starting LLM logical reasoning error pattern experiment...")
    
    client = setup_client()
    
    # Define experimental conditions with equal sample sizes
    conditions = [
        {'complexity': 'simple', 'chain_length': 2, 'operators': ['AND', 'OR']},
        {'complexity': 'simple', 'chain_length': 2, 'operators': ['NOT', 'AND']},
        {'complexity': 'medium', 'chain_length': 3, 'operators': ['AND', 'OR', 'NOT']},
        {'complexity': 'medium', 'chain_length': 3, 'operators': ['OR', 'NOT', 'AND']}
    ]
    
    # Ensure equal sample sizes (minimum 50 total, ~12-13 per condition)
    samples_per_condition = 15  # 60 total samples
    all_results = []
    api_calls_made = 0
    max_api_calls = 30
    
    print(f"Generating {samples_per_condition} samples per condition, {len(conditions)} conditions total")
    
    for cond_idx, condition in enumerate(conditions):
        print(f"Processing condition {cond_idx + 1}/{len(conditions)}: {condition}")
        
        condition_results = []
        
        for sample_idx in range(samples_per_condition):
            if api_calls_made >= max_api_calls:
                print(f"Reached API call limit ({max_api_calls})")
                break
            
            print(f"  Sample {sample_idx + 1}/{samples_per_condition}")
            
            # Generate problem
            problem_data = generate_logical_problem(
                condition['complexity'],
                condition['operators'],
                condition['chain_length']
            )
            
            # Get LLM response
            response = call_llm_with_retry(client, problem_data['problem'])
            api_calls_made += 1
            
            if response is None:
                print("    Skipping due to API failure")
                continue
            
            # Analyze errors
            error_analysis = analyze_llm_response(response, problem_data['steps'])
            
            # Store result
            result = {
                'condition_index': cond_idx,
                'condition': condition,
                'problem_data': problem_data,
                'llm_response': response,
                'error_analysis': error_analysis,
                'api_calls_used': api_calls_made
            }
            
            condition_results.append(result)
            all_results.append(result)
            
            time.sleep(0.1)  # Rate limiting
        
        if api_calls_made >= max_api_calls:
            break
    
    print(f"\nCompleted experiment with {len(all_results)} valid samples and {api_calls_made} API calls")
    
    if len(all_results) < 50:
        print(f"WARNING: Only collected {len(all_results)} samples, below minimum of 50")
    
    return all_results

def analyze_error_patterns(results):
    """Analyze error patterns across logical operators and step positions."""
    print("\nAnalyzing error patterns...")
    
    # Collect error data
    operator_errors = collections.defaultdict(int)
    position_errors = collections.defaultdict(int)
    operator_totals = collections.defaultdict(int)
    position_totals = collections.defaultdict(int)
    
    total_errors = 0
    total_steps = 0
    
    for result in results:
        if result['error_analysis']['parsing_failed']:
            continue
        
        # Count total steps and operators for baseline
        for step in result['problem_data']['steps']:
            operator_totals[step['operator']] += 1
            position_totals[step['position']] += 1
            total_steps += 1
        
        # Count errors by operator and position
        for error in result['error_analysis']['errors']:
            operator_errors[error['operator']] += 1
            position_errors[error['position']] += 1
            total_errors += 1
    
    # Calculate error rates
    operator_error_rates = {}
    for op in ['AND', 'OR', 'NOT']:
        if operator_totals[op] > 0:
            operator_error_rates[op] = float(operator_errors[op] / operator_totals[op])
        else:
            operator_error_rates[op] = 0.0
    
    position_error_rates = {}
    for pos in range(3):  # positions 0, 1, 2
        if position_totals[pos] > 0:
            position_error_rates[pos] = float(position_errors[pos] / position_totals[pos])
        else:
            position_error_rates[pos] = 0.0
    
    # Chi-square test for non-random distribution
    from scipy.stats import chisquare
    
    # Test operator error distribution
    operator_observed = [operator_errors[op] for op in ['AND', 'OR', 'NOT']]
    operator_expected = [operator_totals[op] * (total_errors / total_steps) if total_steps > 0 else 1 
                        for op in ['AND', 'OR', 'NOT']]
    
    try:
        chi2_op, p_op = chisquare(operator_observed, operator_expected)
        chi2_op = float(chi2_op)
        p_op = float(p_op)
    except:
        chi2_op, p_op = 0.0, 1.0
    
    # Test position error distribution  
    position_observed = [position_errors[pos] for pos in range(3)]
    position_expected = [position_totals[pos] * (total_errors / total_steps) if total_steps > 0 else 1 
                        for pos in range(3)]
    
    try:
        chi2_pos, p_pos = chisquare(position_observed, position_expected)
        chi2_pos = float(chi2_pos)
        p_pos = float(p_pos)
    except:
        chi2_pos, p_pos = 0.0, 1.0
    
    # Calculate effect sizes (Cramér's V)
    n_samples = len(results)
    if n_samples > 0:
        cramers_v_op = float(np.sqrt(chi2_op / (n_samples * 2))) if n_samples > 0 else 0.0  # 3-1=2 degrees of freedom
        cramers_v_pos = float(np.sqrt(chi2_pos / (n_samples * 2))) if n_samples > 0 else 0.0  # 3-1=2 degrees of freedom
    else:
        cramers_v_op = 0.0
        cramers_v_pos = 0.0
    
    return {
        'operator_error_rates': operator_error_rates,
        'position_error_rates': position_error_rates,
        'operator_chi2': chi2_op,
        'operator_p_value': p_op,
        'position_chi2': chi2_pos,
        'position_p_value': p_pos,
        'cramers_v_operator': cramers_v_op,
        'cramers_v_position': cramers_v_pos,
        'total_errors': total_errors,
        'total_steps': total_steps,
        'error_pattern_consistency_score': float(max(cramers_v_op, cramers_v_pos))
    }

def save_results(results, analysis, filename):
    """Save results to JSON file with proper serialization."""
    print(f"\nSaving results to {filename}")
    
    # Convert numpy types to Python types for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(item) for item in obj]
        return obj
    
    output_data = {
        'experiment_metadata': {
            'total_samples': len(results),
            'hypothesis': 'LLMs will show consistent patterns in their reasoning errors when solving multi-step logical problems',
            'conditions_tested': 4,
            'min_sample_size_met': len(results) >= 50
        },
        'results': convert_for_json(results),
        'analysis': convert_for_json(analysis)
    }
    
    try:
        with open(filename, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        print(f"Results saved successfully to {filename}")
    except Exception as e:
        print(f"Error saving results: {e}")

def print_results_table(analysis, results):
    """Print a clear results summary table."""
    print("\n" + "="*80)
    print("EXPERIMENT RESULTS SUMMARY")
    print("="*80)
    
    print(f"Total samples collected: {len(results)}")
    print(f"Minimum sample size (50) met: {len(results) >= 50}")
    print(f"Total errors detected: {analysis['total_errors']}")
    print(f"Total reasoning steps: {analysis['total_steps']}")
    
    print("\nERROR RATES BY LOGICAL OPERATOR:")
    print("-" * 40)
    for operator, rate in analysis['operator_error_rates'].items():
        print(f"{operator:>4}: {rate:.3f} ({rate*100:.1f}%)")
    
    print("\nERROR RATES BY STEP POSITION:")
    print("-" * 40)
    for position, rate in analysis['position_error_rates'].items():
        print(f"Pos {position}: {rate:.3f} ({rate*100:.1f}%)")
    
    print("\nSTATISTICAL TESTS:")
    print("-" * 40)
    print(f"Operator clustering - Chi²: {analysis['operator_chi2']:.3f}, p-value: {analysis['operator_p_value']:.3f}")
    print(f"Position clustering - Chi²: {analysis['position_chi2']:.3f}, p-value: {analysis['position_p_value']:.3f}")
    print(f"Cramér's V (operator): {analysis['cramers_v_operator']:.3f}")
    print(f"Cramér's V (position): {analysis['cramers_v_position']:.3f}")
    
    print("\nHYPOTHESIS TESTING:")
    print("-" * 40)
    significant = (analysis['operator_p_value'] < 0.05 or analysis['position_p_value'] < 0.05)
    effect_size_meaningful = (analysis['cramers_v_operator'] > 0.1 or analysis['cramers_v_position'] > 0.1)
    
    print(f"Statistical significance (p < 0.05): {significant}")
    print(f"Meaningful effect size (Cramér's V > 0.1): {effect_size_meaningful}")
    print(f"Hypothesis supported: {significant and effect_size_meaningful}")
    print(f"Error pattern consistency score: {analysis['error_pattern_consistency_score']:.3f}")
    
    print("="*80)

def main():
    """Main experiment execution."""
    try:
        start_time = time.time()
        
        # Run experiment
        results = run_experiment()
        
        if len(results) < 10:  # Absolute minimum for any analysis
            print(f"ERROR: Too few results ({len(results)}) to perform meaningful analysis")
            sys.exit(1)
        
        # Analyze results
        analysis = analyze_error_patterns(results)
        
        # Save results
        output_file = "/Users/kumacmini/cost-aware-research-search/results/iter_03_results.json"
        save_results(results, analysis, output_file)
        
        # Print summary
        print_results_table(analysis, results)
        
        elapsed_time = time.time() - start_time
        print(f"\nExperiment completed in {elapsed_time:.1f} seconds")
        
    except Exception as e:
        print(f"Experiment failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()