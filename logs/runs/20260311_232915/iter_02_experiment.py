import json
import os
import sys
import time
import math
import random
import re
import collections
import statistics
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

def generate_arithmetic_problem():
    """Generate a simple arithmetic problem with 3-7 steps"""
    start_val = random.randint(1, 20)
    n_steps = random.randint(3, 7)
    
    operations = ['+', '-', '*']
    problem_steps = []
    current_val = start_val
    
    for i in range(n_steps):
        op = random.choice(operations)
        operand = random.randint(1, 10)
        
        if op == '+':
            next_val = current_val + operand
        elif op == '-':
            next_val = current_val - operand
        else:  # multiplication
            next_val = current_val * operand
            
        problem_steps.append({
            'step_num': i + 1,
            'prev_val': current_val,
            'operation': op,
            'operand': operand,
            'next_val': next_val
        })
        current_val = next_val
    
    return start_val, problem_steps

def generate_counting_problem():
    """Generate a counting problem with increments/decrements"""
    start_count = random.randint(0, 15)
    n_steps = random.randint(3, 6)
    
    problem_steps = []
    current_count = start_count
    
    for i in range(n_steps):
        change = random.choice([-3, -2, -1, 1, 2, 3, 4, 5])
        next_count = max(0, current_count + change)
        
        problem_steps.append({
            'step_num': i + 1,
            'prev_val': current_count,
            'operation': 'count',
            'operand': change,
            'next_val': next_count
        })
        current_count = next_count
    
    return start_count, problem_steps

def render_trace(start_val, steps, problem_type="arithmetic"):
    """Render problem steps as text trace"""
    lines = [f"Starting with {start_val}"]
    
    for step in steps:
        if problem_type == "arithmetic":
            if step['operation'] == '+':
                line = f"Step {step['step_num']}: {step['prev_val']} + {step['operand']} = {step['next_val']}"
            elif step['operation'] == '-':
                line = f"Step {step['step_num']}: {step['prev_val']} - {step['operand']} = {step['next_val']}"
            else:  # multiplication
                line = f"Step {step['step_num']}: {step['prev_val']} * {step['operand']} = {step['next_val']}"
        else:  # counting
            if step['operand'] >= 0:
                line = f"Step {step['step_num']}: Count {step['prev_val']} + {step['operand']} = {step['next_val']}"
            else:
                line = f"Step {step['step_num']}: Count {step['prev_val']} - {abs(step['operand'])} = {step['next_val']}"
        
        lines.append(line)
    
    lines.append(f"Final answer: {steps[-1]['next_val']}")
    return lines

def inject_corruption(start_val, steps, corruption_type, error_position="middle"):
    """Inject various types of corruption into reasoning traces"""
    corrupted_steps = [step.copy() for step in steps]
    
    if error_position == "early":
        target_idx = 0 if len(steps) > 1 else 0
    elif error_position == "late":
        target_idx = len(steps) - 1
    else:  # middle
        target_idx = len(steps) // 2
    
    if corruption_type == "operator_swap":
        # Change operation but keep operand
        if corrupted_steps[target_idx]['operation'] == '+':
            corrupted_steps[target_idx]['operation'] = '-'
        elif corrupted_steps[target_idx]['operation'] == '-':
            corrupted_steps[target_idx]['operation'] = '+'
        else:  # multiplication
            corrupted_steps[target_idx]['operation'] = '+'
    
    elif corruption_type == "sign_flip":
        # Flip sign of operand
        corrupted_steps[target_idx]['operand'] = -corrupted_steps[target_idx]['operand']
    
    elif corruption_type == "off_by_one":
        # Add or subtract 1 from result
        corrupted_steps[target_idx]['next_val'] += random.choice([-1, 1])
    
    elif corruption_type == "stale_intermediate":
        # Reuse a previous intermediate value
        if target_idx > 0:
            corrupted_steps[target_idx]['next_val'] = corrupted_steps[target_idx-1]['next_val']
    
    elif corruption_type == "compensating_error":
        # Make two errors that partially cancel out
        if target_idx < len(steps) - 1:
            # First error: add extra amount
            extra = random.randint(2, 5)
            corrupted_steps[target_idx]['next_val'] += extra
            # Second error: subtract similar amount
            corrupted_steps[target_idx + 1]['next_val'] -= (extra - random.randint(-1, 1))
    
    # Recalculate downstream values for consistency
    for i in range(target_idx + 1, len(corrupted_steps)):
        prev_val = corrupted_steps[i-1]['next_val']
        corrupted_steps[i]['prev_val'] = prev_val
        
        op = corrupted_steps[i]['operation']
        operand = corrupted_steps[i]['operand']
        
        if op == '+':
            corrupted_steps[i]['next_val'] = prev_val + operand
        elif op == '-':
            corrupted_steps[i]['next_val'] = prev_val - operand
        elif op == '*':
            corrupted_steps[i]['next_val'] = prev_val * operand
        elif op == 'count':
            corrupted_steps[i]['next_val'] = max(0, prev_val + operand)
    
    return corrupted_steps

def line_local_checker(trace_lines):
    """Baseline checker: validate arithmetic in each line + final answer"""
    errors = []
    
    for i, line in enumerate(trace_lines[1:-1], 1):  # Skip start and final answer
        if "=" in line:
            parts = line.split("=")
            if len(parts) == 2:
                try:
                    left_part = parts[0].strip()
                    expected_result = float(parts[1].strip())
                    
                    # Extract arithmetic expression
                    if "+" in left_part:
                        expr_parts = left_part.split("+")
                        if len(expr_parts) == 2:
                            a = float(expr_parts[0].split()[-1])
                            b = float(expr_parts[1].strip())
                            actual_result = a + b
                        else:
                            continue
                    elif "-" in left_part and left_part.count("-") == 1:
                        expr_parts = left_part.split("-")
                        if len(expr_parts) == 2:
                            a = float(expr_parts[0].split()[-1])
                            b = float(expr_parts[1].strip())
                            actual_result = a - b
                        else:
                            continue
                    elif "*" in left_part:
                        expr_parts = left_part.split("*")
                        if len(expr_parts) == 2:
                            a = float(expr_parts[0].split()[-1])
                            b = float(expr_parts[1].strip())
                            actual_result = a * b
                        else:
                            continue
                    else:
                        continue
                    
                    if abs(actual_result - expected_result) > 0.001:
                        errors.append(i)
                
                except (ValueError, IndexError):
                    errors.append(i)
    
    return len(errors) == 0

def state_replay_verifier(trace_lines):
    """Global checker: replay full state across entire trace"""
    try:
        # Extract starting value
        start_line = trace_lines[0]
        start_val = float(start_line.split()[-1])
        
        current_state = start_val
        
        # Process each step and maintain global state
        for i, line in enumerate(trace_lines[1:-1], 1):
            if "=" in line:
                parts = line.split("=")
                if len(parts) != 2:
                    return False
                
                expected_result = float(parts[1].strip())
                left_part = parts[0].strip()
                
                # Extract operation and operand
                if "+" in left_part:
                    expr_parts = left_part.split("+")
                    if len(expr_parts) == 2:
                        stated_prev = float(expr_parts[0].split()[-1])
                        operand = float(expr_parts[1].strip())
                        
                        # Check state consistency
                        if abs(current_state - stated_prev) > 0.001:
                            return False
                        
                        # Update state
                        current_state = current_state + operand
                        
                        # Check computation
                        if abs(current_state - expected_result) > 0.001:
                            return False
                    else:
                        return False
                        
                elif "-" in left_part and left_part.count("-") == 1:
                    expr_parts = left_part.split("-")
                    if len(expr_parts) == 2:
                        stated_prev = float(expr_parts[0].split()[-1])
                        operand = float(expr_parts[1].strip())
                        
                        if abs(current_state - stated_prev) > 0.001:
                            return False
                        
                        current_state = current_state - operand
                        
                        if abs(current_state - expected_result) > 0.001:
                            return False
                    else:
                        return False
                        
                elif "*" in left_part:
                    expr_parts = left_part.split("*")
                    if len(expr_parts) == 2:
                        stated_prev = float(expr_parts[0].split()[-1])
                        operand = float(expr_parts[1].strip())
                        
                        if abs(current_state - stated_prev) > 0.001:
                            return False
                        
                        current_state = current_state * operand
                        
                        if abs(current_state - expected_result) > 0.001:
                            return False
                    else:
                        return False
                else:
                    return False
        
        # Check final answer consistency
        final_line = trace_lines[-1]
        final_answer = float(final_line.split()[-1])
        
        return abs(current_state - final_answer) < 0.001
        
    except (ValueError, IndexError):
        return False

def generate_dataset(n_samples=1320):
    """Generate complete dataset with multiple conditions"""
    
    conditions = [
        "clean",
        "operator_swap_early", "operator_swap_middle", "operator_swap_late",
        "sign_flip_early", "sign_flip_middle", "sign_flip_late", 
        "off_by_one_middle", "stale_intermediate_middle",
        "compensating_error_early", "compensating_error_middle"
    ]
    
    dataset = []
    samples_per_condition = n_samples // len(conditions)
    
    print(f"Generating {n_samples} samples across {len(conditions)} conditions...")
    
    for cond_idx, condition in enumerate(conditions):
        print(f"Processing condition {cond_idx+1}/{len(conditions)}: {condition}")
        
        for sample_idx in range(samples_per_condition):
            if sample_idx % 20 == 0:
                print(f"  Sample {sample_idx+1}/{samples_per_condition}")
            
            # Generate base problem
            problem_type = random.choice(["arithmetic", "counting"])
            
            if problem_type == "arithmetic":
                start_val, steps = generate_arithmetic_problem()
            else:
                start_val, steps = generate_counting_problem()
            
            if condition == "clean":
                final_steps = steps
                is_corrupted = False
            else:
                # Parse corruption type and position
                corruption_parts = condition.split("_")
                corruption_type = "_".join(corruption_parts[:-1])
                error_position = corruption_parts[-1]
                
                final_steps = inject_corruption(start_val, steps, corruption_type, error_position)
                is_corrupted = True
            
            trace_lines = render_trace(start_val, final_steps, problem_type)
            
            # Apply both checkers
            line_local_result = line_local_checker(trace_lines)
            state_replay_result = state_replay_verifier(trace_lines)
            
            sample = {
                'condition': condition,
                'problem_type': problem_type,
                'is_corrupted': is_corrupted,
                'trace_lines': trace_lines,
                'line_local_correct': line_local_result,
                'state_replay_correct': state_replay_result,
                'start_val': float(start_val),
                'n_steps': len(final_steps)
            }
            
            dataset.append(sample)
    
    print(f"Generated {len(dataset)} total samples")
    return dataset

def evaluate_checkers(dataset):
    """Evaluate both checkers and compute metrics"""
    
    # Prepare data for evaluation
    y_true = [not sample['is_corrupted'] for sample in dataset]  # True if trace is valid
    y_pred_local = [sample['line_local_correct'] for sample in dataset]
    y_pred_replay = [sample['state_replay_correct'] for sample in dataset]
    
    # Overall metrics
    local_precision, local_recall, local_f1, _ = precision_recall_fscore_support(
        y_true, y_pred_local, average='macro', zero_division=0
    )
    
    replay_precision, replay_recall, replay_f1, _ = precision_recall_fscore_support(
        y_true, y_pred_replay, average='macro', zero_division=0
    )
    
    # Per-condition analysis
    condition_results = {}
    conditions = list(set(sample['condition'] for sample in dataset))
    
    for condition in conditions:
        condition_samples = [s for s in dataset if s['condition'] == condition]
        
        cond_y_true = [not s['is_corrupted'] for s in condition_samples]
        cond_y_local = [s['line_local_correct'] for s in condition_samples]
        cond_y_replay = [s['state_replay_correct'] for s in condition_samples]
        
        if len(set(cond_y_true)) > 1:  # Only if both classes present
            local_prec, local_rec, local_f1_cond, _ = precision_recall_fscore_support(
                cond_y_true, cond_y_local, average='macro', zero_division=0
            )
            replay_prec, replay_rec, replay_f1_cond, _ = precision_recall_fscore_support(
                cond_y_true, cond_y_replay, average='macro', zero_division=0
            )
        else:
            local_prec = local_rec = local_f1_cond = 0.0
            replay_prec = replay_rec = replay_f1_cond = 0.0
        
        condition_results[condition] = {
            'n_samples': len(condition_samples),
            'local_precision': float(local_prec),
            'local_recall': float(local_rec),
            'local_f1': float(local_f1_cond),
            'replay_precision': float(replay_prec),
            'replay_recall': float(replay_rec),
            'replay_f1': float(replay_f1_cond),
            'f1_improvement': float(replay_f1_cond - local_f1_cond),
            'recall_improvement': float(replay_rec - local_rec)
        }
    
    return {
        'overall_local_precision': float(local_precision),
        'overall_local_recall': float(local_recall),
        'overall_local_f1': float(local_f1),
        'overall_replay_precision': float(replay_precision),
        'overall_replay_recall': float(replay_recall),
        'overall_replay_f1': float(replay_f1),
        'overall_f1_improvement': float(replay_f1 - local_f1),
        'overall_recall_improvement': float(replay_recall - local_recall),
        'condition_results': condition_results
    }

def robustness_analysis(dataset):
    """Additional robustness analyses with different approaches"""
    print("Running robustness analyses...")
    
    # Test with different text representations
    vectorizers = {
        'tfidf': TfidfVectorizer(max_features=100, ngram_range=(1, 2)),
        'count': CountVectorizer(max_features=100, ngram_range=(1, 2)),
        'char_ngrams': TfidfVectorizer(analyzer='char', ngram_range=(2, 4), max_features=50)
    }
    
    robustness_results = {}
    
    # Analyze trace length effects
    short_traces = [s for s in dataset if s['n_steps'] <= 4]
    long_traces = [s for s in dataset if s['n_steps'] > 4]
    
    for trace_type, traces in [("short", short_traces), ("long", long_traces)]:
        if len(traces) > 10:
            y_true = [not s['is_corrupted'] for s in traces]
            y_local = [s['line_local_correct'] for s in traces]
            y_replay = [s['state_replay_correct'] for s in traces]
            
            if len(set(y_true)) > 1:
                _, _, local_f1, _ = precision_recall_fscore_support(y_true, y_local, average='macro', zero_division=0)
                _, _, replay_f1, _ = precision_recall_fscore_support(y_true, y_replay, average='macro', zero_division=0)
                
                robustness_results[f'{trace_type}_traces'] = {
                    'n_samples': len(traces),
                    'local_f1': float(local_f1),
                    'replay_f1': float(replay_f1),
                    'improvement': float(replay_f1 - local_f1)
                }
    
    # Test multiple random seeds
    seed_results = []
    original_seed = np.random.get_state()
    
    for seed in [123, 456, 789]:
        np.random.seed(seed)
        random.seed(seed)
        
        # Generate smaller subset for speed
        subset = generate_dataset(n_samples=200)
        results = evaluate_checkers(subset)
        seed_results.append(results['overall_f1_improvement'])
    
    # Restore original seed
    np.random.set_state(original_seed)
    
    robustness_results['seed_stability'] = {
        'improvements': [float(x) for x in seed_results],
        'mean_improvement': float(np.mean(seed_results)),
        'std_improvement': float(np.std(seed_results))
    }
    
    return robustness_results

def main():
    start_time = time.time()
    
    print("Starting LLM Step-by-Step Reasoning Trace Verification Experiment")
    print("=" * 70)
    
    # Generate main dataset
    dataset = generate_dataset(n_samples=1320)
    
    print(f"\nEvaluating {len(dataset)} samples...")
    
    # Main evaluation
    results = evaluate_checkers(dataset)
    
    # Robustness analysis
    robustness = robustness_analysis(dataset)
    
    # Compile final results
    final_results = {
        'experiment': 'llm_reasoning_trace_verification',
        'n_samples': len(dataset),
        'n_conditions': len(set(s['condition'] for s in dataset)),
        'main_results': results,
        'robustness_analysis': robustness,
        'success_criteria': {
            'f1_improvement_threshold': 0.15,
            'recall_improvement_threshold': 0.30,
            'f1_improvement_achieved': results['overall_f1_improvement'],
            'recall_improvement_achieved': results['overall_recall_improvement'],
            'f1_success': results['overall_f1_improvement'] >= 0.15,
            'recall_success': results['overall_recall_improvement'] >= 0.30
        },
        'runtime_seconds': float(time.time() - start_time)
    }
    
    # Save results
    os.makedirs('/Users/kumacmini/cost-aware-research-search/results', exist_ok=True)
    
    with open('/Users/kumacmini/cost-aware-research-search/results/iter_02_results.json', 'w') as f:
        json.dump(final_results, f, indent=2, default=str)
    
    # Print results table
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)
    print(f"Total Samples: {len(dataset)}")
    print(f"Conditions: {len(set(s['condition'] for s in dataset))}")
    print(f"Runtime: {time.time() - start_time:.1f} seconds")
    print()
    
    print("OVERALL PERFORMANCE:")
    print(f"{'Metric':<25} {'Local Checker':<15} {'State Replay':<15} {'Improvement':<15}")
    print("-" * 70)
    print(f"{'Precision':<25} {results['overall_local_precision']:<15.3f} {results['overall_replay_precision']:<15.3f} {results['overall_replay_precision'] - results['overall_local_precision']:<15.3f}")
    print(f"{'Recall':<25} {results['overall_local_recall']:<15.3f} {results['overall_replay_recall']:<15.3f} {results['overall_recall_improvement']:<15.3f}")
    print(f"{'Macro F1':<25} {results['overall_local_f1']:<15.3f} {results['overall_replay_f1']:<15.3f} {results['overall_f1_improvement']:<15.3f}")
    print()
    
    print("PER-CONDITION RESULTS:")
    print(f"{'Condition':<25} {'Samples':<10} {'Local F1':<10} {'Replay F1':<10} {'F1 Δ':<10} {'Recall Δ':<10}")
    print("-" * 75)
    
    for condition, cond_results in sorted(results['condition_results'].items()):
        print(f"{condition:<25} {cond_results['n_samples']:<10} {cond_results['local_f1']:<10.3f} {cond_results['replay_f1']:<10.3f} {cond_results['f1_improvement']:<10.3f} {cond_results['recall_improvement']:<10.3f}")
    
    print()
    print("SUCCESS CRITERIA:")
    print(f"F1 Improvement ≥ 0.15: {'✓' if final_results['success_criteria']['f1_success'] else '✗'} ({results['overall_f1_improvement']:.3f})")
    print(f"Recall Improvement ≥ 0.30: {'✓' if final_results['success_criteria']['recall_success'] else '✗'} ({results['overall_recall_improvement']:.3f})")
    
    if 'seed_stability' in robustness:
        print(f"\nSeed Stability: μ={robustness['seed_stability']['mean_improvement']:.3f}, σ={robustness['seed_stability']['std_improvement']:.3f}")
    
    print("\nExperiment completed successfully!")

if __name__ == "__main__":
    main()