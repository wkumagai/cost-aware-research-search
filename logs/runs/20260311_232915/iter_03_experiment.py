import json
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from statistics import mean, stdev
import math
import os
import time
import re

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

class StringTransformationRule:
    def __init__(self, rule_type, params):
        self.rule_type = rule_type
        self.params = params
    
    def apply(self, input_str):
        if self.rule_type == "prefix":
            return self.params["prefix"] + input_str
        elif self.rule_type == "suffix":
            return input_str + self.params["suffix"]
        elif self.rule_type == "replace_char":
            return input_str.replace(self.params["from_char"], self.params["to_char"])
        elif self.rule_type == "uppercase_nth":
            n = self.params["position"]
            if len(input_str) > n:
                return input_str[:n] + input_str[n].upper() + input_str[n+1:]
            return input_str
        elif self.rule_type == "reverse_substring":
            start, end = self.params["start"], self.params["end"]
            if len(input_str) > end:
                return input_str[:start] + input_str[start:end+1][::-1] + input_str[end+1:]
            return input_str
        return input_str

def generate_rule_pair():
    """Generate a target rule and a similar distractor rule"""
    rule_families = [
        ("prefix", [
            {"prefix": "pre_"}, {"prefix": "pro_"}
        ]),
        ("suffix", [
            {"suffix": "_end"}, {"suffix": "_fin"}
        ]),
        ("replace_char", [
            {"from_char": "a", "to_char": "x"}, {"from_char": "a", "to_char": "y"}
        ]),
        ("uppercase_nth", [
            {"position": 1}, {"position": 2}
        ]),
        ("reverse_substring", [
            {"start": 1, "end": 3}, {"start": 0, "end": 2}
        ])
    ]
    
    family_name, rule_configs = random.choice(rule_families)
    target_config, distractor_config = rule_configs
    
    target_rule = StringTransformationRule(family_name, target_config)
    distractor_rule = StringTransformationRule(family_name, distractor_config)
    
    return target_rule, distractor_rule, family_name

def generate_test_inputs():
    """Generate diverse test input strings"""
    base_words = ["cat", "dog", "car", "sun", "box", "key", "map", "pen", "cup", "hat"]
    variants = []
    
    for word in base_words[:6]:
        variants.extend([
            word,
            word + "s",
            word.upper(),
            word + word[0],  # repeat first char
            "a" + word,      # add prefix
            word + "z"       # add suffix
        ])
    
    return variants

def create_ambiguous_examples(target_rule, distractor_rule, test_inputs, n_examples=3):
    """Create examples that are compatible with both rules"""
    ambiguous_examples = []
    used_inputs = set()
    
    for _ in range(n_examples):
        attempts = 0
        while attempts < 50:
            input_str = random.choice(test_inputs)
            if input_str in used_inputs:
                attempts += 1
                continue
                
            target_output = target_rule.apply(input_str)
            distractor_output = distractor_rule.apply(input_str)
            
            # Check if outputs are the same (ambiguous)
            if target_output == distractor_output:
                ambiguous_examples.append((input_str, target_output))
                used_inputs.add(input_str)
                break
            attempts += 1
    
    return ambiguous_examples

def create_diagnostic_example(target_rule, distractor_rule, test_inputs, used_inputs):
    """Create an example that disambiguates between the rules"""
    for _ in range(100):
        input_str = random.choice(test_inputs)
        if input_str in used_inputs:
            continue
            
        target_output = target_rule.apply(input_str)
        distractor_output = distractor_rule.apply(input_str)
        
        if target_output != distractor_output:
            return (input_str, target_output)
    
    # Fallback: create a synthetic diagnostic example
    synthetic_input = "test_" + str(random.randint(100, 999))
    target_output = target_rule.apply(synthetic_input)
    return (synthetic_input, target_output)

def simulate_model_prediction(prompt, query_input, target_rule, distractor_rule, recency_weight=0.3):
    """Simulate a model's prediction with recency bias"""
    # Extract examples from prompt
    examples = []
    lines = prompt.strip().split('\n')
    
    for line in lines:
        if '->' in line:
            parts = line.split('->')
            if len(parts) == 2:
                input_part = parts[0].strip()
                output_part = parts[1].strip()
                examples.append((input_part, output_part))
    
    if not examples:
        return distractor_rule.apply(query_input)
    
    # Calculate weighted vote with recency bias
    target_score = 0
    distractor_score = 0
    
    for i, (ex_input, ex_output) in enumerate(examples):
        # Higher weight for later examples
        weight = 1 + recency_weight * i
        
        target_pred = target_rule.apply(ex_input)
        distractor_pred = distractor_rule.apply(ex_input)
        
        if ex_output == target_pred:
            target_score += weight
        if ex_output == distractor_pred:
            distractor_score += weight
    
    # Add some noise
    noise = random.uniform(-0.1, 0.1)
    target_score += noise
    
    if target_score > distractor_score:
        return target_rule.apply(query_input)
    else:
        return distractor_rule.apply(query_input)

def create_prompt_conditions(ambiguous_examples, diagnostic_example):
    """Create different prompt orderings and formats"""
    all_examples = ambiguous_examples + [diagnostic_example]
    
    conditions = {}
    
    # 1. Random order (baseline)
    shuffled = all_examples.copy()
    random.shuffle(shuffled)
    conditions["random_order"] = format_examples(shuffled)
    
    # 2. Diagnostic first
    diagnostic_first = [diagnostic_example] + ambiguous_examples
    conditions["diagnostic_first"] = format_examples(diagnostic_first)
    
    # 3. Diagnostic last
    diagnostic_last = ambiguous_examples + [diagnostic_example]
    conditions["diagnostic_last"] = format_examples(diagnostic_last)
    
    # 4. Labeled diagnostic block
    labeled_prompt = format_examples(ambiguous_examples) + "\n\nKey Example:\n" + format_examples([diagnostic_example])
    conditions["labeled_diagnostic"] = labeled_prompt
    
    # 5. Control with extra formatting but diagnostic not special
    control_shuffled = all_examples.copy()
    random.shuffle(control_shuffled)
    first_part = control_shuffled[:-1]
    last_part = control_shuffled[-1:]
    control_prompt = format_examples(first_part) + "\n\nAdditional Example:\n" + format_examples(last_part)
    conditions["control_formatting"] = control_prompt
    
    return conditions

def format_examples(examples):
    """Format examples into prompt text"""
    lines = []
    for input_str, output_str in examples:
        lines.append(f"{input_str} -> {output_str}")
    return '\n'.join(lines)

def evaluate_sample(sample_data, recency_weight=0.3):
    """Evaluate a single sample across all conditions"""
    results = {}
    target_rule = sample_data["target_rule"]
    distractor_rule = sample_data["distractor_rule"]
    query_input = sample_data["query_input"]
    correct_output = sample_data["correct_output"]
    
    for condition_name, prompt in sample_data["conditions"].items():
        predicted_output = simulate_model_prediction(
            prompt, query_input, target_rule, distractor_rule, recency_weight
        )
        is_correct = predicted_output == correct_output
        results[condition_name] = {
            "predicted": predicted_output,
            "correct": correct_output,
            "is_correct": is_correct
        }
    
    return results

def bootstrap_confidence_interval(data, n_bootstrap=1000, confidence=0.95):
    """Calculate bootstrap confidence interval"""
    if not data:
        return 0.0, 0.0, 0.0
    
    bootstrap_means = []
    n = len(data)
    
    for _ in range(n_bootstrap):
        sample = [data[random.randint(0, n-1)] for _ in range(n)]
        bootstrap_means.append(mean(sample))
    
    bootstrap_means.sort()
    alpha = 1 - confidence
    lower_idx = int(alpha/2 * n_bootstrap)
    upper_idx = int((1 - alpha/2) * n_bootstrap)
    
    return mean(data), bootstrap_means[lower_idx], bootstrap_means[upper_idx]

def calculate_confusion_matrix(predictions, actuals):
    """Calculate confusion matrix metrics"""
    tp = fp = tn = fn = 0
    
    for pred, actual in zip(predictions, actuals):
        if pred and actual:
            tp += 1
        elif pred and not actual:
            fp += 1
        elif not pred and actual:
            fn += 1
        else:
            tn += 1
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0.0
    
    return {
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy)
    }

def run_experiment():
    """Run the complete experiment"""
    print("Starting Few-Shot Prompt Ordering Experiment")
    print("=" * 50)
    
    n_samples = 400
    recency_weights = [0.1, 0.3, 0.5]
    n_seeds = 5
    
    all_results = []
    test_inputs = generate_test_inputs()
    
    # Generate samples
    print("Generating samples...")
    samples = []
    
    for i in range(n_samples):
        if i % 50 == 0:
            print(f"Generating sample {i+1}/{n_samples}...")
        
        # Generate rules and examples
        target_rule, distractor_rule, rule_family = generate_rule_pair()
        
        # Create ambiguous examples
        ambiguous_examples = create_ambiguous_examples(target_rule, distractor_rule, test_inputs, 3)
        
        # If we couldn't find enough ambiguous examples, create synthetic ones
        while len(ambiguous_examples) < 3:
            synthetic_input = f"syn_{len(ambiguous_examples)}_{i}"
            # Force ambiguity by using target output
            synthetic_output = target_rule.apply(synthetic_input)
            ambiguous_examples.append((synthetic_input, synthetic_output))
        
        used_inputs = {ex[0] for ex in ambiguous_examples}
        diagnostic_example = create_diagnostic_example(target_rule, distractor_rule, test_inputs, used_inputs)
        
        # Create query
        query_input = f"query_{i}"
        while query_input in used_inputs:
            query_input = f"query_{i}_{random.randint(0, 999)}"
        correct_output = target_rule.apply(query_input)
        
        # Create prompt conditions
        conditions = create_prompt_conditions(ambiguous_examples, diagnostic_example)
        
        sample_data = {
            "sample_id": i,
            "rule_family": rule_family,
            "target_rule": target_rule,
            "distractor_rule": distractor_rule,
            "ambiguous_examples": ambiguous_examples,
            "diagnostic_example": diagnostic_example,
            "query_input": query_input,
            "correct_output": correct_output,
            "conditions": conditions
        }
        
        samples.append(sample_data)
    
    print(f"Generated {len(samples)} samples")
    
    # Run evaluations
    print("\nRunning evaluations...")
    
    for seed_idx in range(n_seeds):
        print(f"\nSeed {seed_idx + 1}/{n_seeds}")
        random.seed(42 + seed_idx)
        np.random.seed(42 + seed_idx)
        
        for weight_idx, recency_weight in enumerate(recency_weights):
            print(f"  Recency weight {recency_weight} ({weight_idx + 1}/{len(recency_weights)})")
            
            condition_results = defaultdict(list)
            rule_family_results = defaultdict(lambda: defaultdict(list))
            
            for sample_idx, sample_data in enumerate(samples):
                if sample_idx % 100 == 0 and sample_idx > 0:
                    print(f"    Processing sample {sample_idx}/{len(samples)}...")
                
                sample_results = evaluate_sample(sample_data, recency_weight)
                
                for condition, result in sample_results.items():
                    condition_results[condition].append(result["is_correct"])
                    rule_family_results[sample_data["rule_family"]][condition].append(result["is_correct"])
            
            # Calculate metrics for this seed and weight
            seed_result = {
                "seed": seed_idx,
                "recency_weight": float(recency_weight),
                "condition_metrics": {},
                "rule_family_metrics": {},
                "n_samples": len(samples)
            }
            
            for condition, correct_list in condition_results.items():
                if correct_list:  # Ensure we have data
                    accuracy = mean(correct_list)
                    confusion = calculate_confusion_matrix(correct_list, [True] * len(correct_list))
                    
                    seed_result["condition_metrics"][condition] = {
                        "accuracy": float(accuracy),
                        "n_samples": len(correct_list),
                        "confusion_matrix": confusion
                    }
            
            for rule_family, rule_conditions in rule_family_results.items():
                seed_result["rule_family_metrics"][rule_family] = {}
                for condition, correct_list in rule_conditions.items():
                    if correct_list:
                        accuracy = mean(correct_list)
                        seed_result["rule_family_metrics"][rule_family][condition] = {
                            "accuracy": float(accuracy),
                            "n_samples": len(correct_list)
                        }
            
            all_results.append(seed_result)
    
    print(f"\nCompleted evaluation with {len(all_results)} result sets")
    
    # Aggregate results
    print("Aggregating results...")
    
    aggregated_results = {}
    
    for recency_weight in recency_weights:
        weight_results = [r for r in all_results if r["recency_weight"] == recency_weight]
        
        condition_aggregates = {}
        conditions = set()
        for result in weight_results:
            conditions.update(result["condition_metrics"].keys())
        
        for condition in conditions:
            accuracies = []
            total_samples = 0
            confusion_matrices = []
            
            for result in weight_results:
                if condition in result["condition_metrics"]:
                    accuracies.append(result["condition_metrics"][condition]["accuracy"])
                    total_samples = result["condition_metrics"][condition]["n_samples"]
                    confusion_matrices.append(result["condition_metrics"][condition]["confusion_matrix"])
            
            if accuracies:
                mean_acc, ci_lower, ci_upper = bootstrap_confidence_interval(accuracies)
                
                # Aggregate confusion matrix
                if confusion_matrices:
                    agg_confusion = {
                        "tp": int(mean([cm["tp"] for cm in confusion_matrices])),
                        "fp": int(mean([cm["fp"] for cm in confusion_matrices])),
                        "tn": int(mean([cm["tn"] for cm in confusion_matrices])),
                        "fn": int(mean([cm["fn"] for cm in confusion_matrices])),
                        "precision": float(mean([cm["precision"] for cm in confusion_matrices])),
                        "recall": float(mean([cm["recall"] for cm in confusion_matrices])),
                        "f1": float(mean([cm["f1"] for cm in confusion_matrices])),
                        "accuracy": float(mean([cm["accuracy"] for cm in confusion_matrices]))
                    }
                else:
                    agg_confusion = {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0}
                
                condition_aggregates[condition] = {
                    "mean_accuracy": float(mean_acc),
                    "ci_lower": float(ci_lower),
                    "ci_upper": float(ci_upper),
                    "std_accuracy": float(stdev(accuracies)) if len(accuracies) > 1 else 0.0,
                    "n_seeds": len(accuracies),
                    "samples_per_seed": int(total_samples),
                    "confusion_matrix": agg_confusion
                }
        
        aggregated_results[recency_weight] = condition_aggregates
    
    # Create final results structure
    final_results = {
        "experiment_metadata": {
            "hypothesis": "Diagnostic example positioning affects accuracy",
            "n_samples_total": n_samples,
            "n_seeds": n_seeds,
            "recency_weights": recency_weights,
            "conditions": list(conditions),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "aggregated_results": aggregated_results,
        "raw_results": all_results,
        "success_criteria": {
            "threshold": 0.12,
            "baseline_condition": "random_order",
            "target_conditions": ["diagnostic_last", "labeled_diagnostic"]
        }
    }
    
    # Print results table
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)
    
    for weight in recency_weights:
        print(f"\nRecency Weight: {weight}")
        print("-" * 60)
        print(f"{'Condition':<20} {'Accuracy':<10} {'CI Lower':<10} {'CI Upper':<10} {'Std Dev':<10}")
        print("-" * 60)
        
        baseline_acc = 0.0
        if "random_order" in aggregated_results[weight]:
            baseline_acc = aggregated_results[weight]["random_order"]["mean_accuracy"]
        
        for condition, metrics in aggregated_results[weight].items():
            acc = metrics["mean_accuracy"]
            ci_lower = metrics["ci_lower"]
            ci_upper = metrics["ci_upper"]
            std_dev = metrics["std_accuracy"]
            
            improvement = acc - baseline_acc
            marker = " *" if improvement >= 0.12 and condition != "random_order" else ""
            
            print(f"{condition:<20} {acc:<10.3f} {ci_lower:<10.3f} {ci_upper:<10.3f} {std_dev:<10.3f}{marker}")
        
        print(f"\nBaseline (random_order): {baseline_acc:.3f}")
        print("* indicates >=12pp improvement over baseline")
    
    # Check success criteria
    success = False
    best_improvement = 0.0
    
    moderate_weight = 0.3
    if moderate_weight in aggregated_results:
        baseline_acc = aggregated_results[moderate_weight].get("random_order", {}).get("mean_accuracy", 0.0)
        
        for target_condition in ["diagnostic_last", "labeled_diagnostic"]:
            if target_condition in aggregated_results[moderate_weight]:
                target_acc = aggregated_results[moderate_weight][target_condition]["mean_accuracy"]
                improvement = target_acc - baseline_acc
                best_improvement = max(best_improvement, improvement)
                if improvement >= 0.12:
                    success = True
    
    final_results["experiment_success"] = {
        "meets_criteria": success,
        "best_improvement": float(best_improvement),
        "baseline_accuracy": float(baseline_acc) if 'baseline_acc' in locals() else 0.0
    }
    
    print(f"\nEXPERIMENT SUCCESS: {success}")
    print(f"Best improvement over baseline: {best_improvement:.3f}")
    
    # Save results
    output_file = "/Users/kumacmini/cost-aware-research-search/results/iter_03_results.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    try:
        with open(output_file, 'w') as f:
            json.dump(final_results, f, indent=2, default=str)
        print(f"\nResults saved to: {output_file}")
    except Exception as e:
        print(f"Error saving results: {e}")
    
    return final_results

if __name__ == "__main__":
    start_time = time.time()
    results = run_experiment()
    end_time = time.time()
    print(f"\nTotal runtime: {end_time - start_time:.1f} seconds")