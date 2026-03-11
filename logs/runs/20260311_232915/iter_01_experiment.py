import json
import os
import random
import time
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

def generate_object_sequences(n_base=120):
    """Generate base object sequences with controlled templates"""
    print("Generating base object sequences...")
    
    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange']
    shapes = ['circle', 'square', 'triangle', 'diamond', 'star', 'hexagon']
    
    # Symmetric templates that preserve unigram counts under swaps
    templates = [
        "{color1} {shape1} then {color2} {shape2}",
        "{color1} {shape1} followed by {color2} {shape2}",
        "{color1} {shape1} and then {color2} {shape2}"
    ]
    
    sequences = []
    for i in range(n_base):
        if i % 10 == 0:
            print(f"Processing base sequence {i+1}/{n_base}...")
            
        # Select random colors and shapes (ensuring different objects)
        color1, color2 = random.sample(colors, 2)
        shape1, shape2 = random.sample(shapes, 2)
        template = random.choice(templates)
        
        sequence = template.format(
            color1=color1, shape1=shape1,
            color2=color2, shape2=shape2
        )
        sequences.append({
            'text': sequence,
            'color1': color1, 'shape1': shape1,
            'color2': color2, 'shape2': shape2,
            'template': template
        })
    
    return sequences

def create_edits(base_sequences):
    """Create controlled edits: order swaps, color substitutions, shape substitutions"""
    print("Creating controlled edits...")
    
    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange']
    shapes = ['circle', 'square', 'triangle', 'diamond', 'star', 'hexagon']
    
    edits = []
    
    for i, seq in enumerate(base_sequences):
        if i % 30 == 0:
            print(f"Processing edits for sequence {i+1}/{len(base_sequences)}...")
            
        base_text = seq['text']
        
        # Order swap (preserves multiset)
        swap_text = seq['template'].format(
            color1=seq['color2'], shape1=seq['shape2'],
            color2=seq['color1'], shape2=seq['shape1']
        )
        
        # Color substitution (first color)
        new_color1 = random.choice([c for c in colors if c != seq['color1']])
        color_sub_text = seq['template'].format(
            color1=new_color1, shape1=seq['shape1'],
            color2=seq['color2'], shape2=seq['shape2']
        )
        
        # Shape substitution (first shape)
        new_shape1 = random.choice([s for s in shapes if s != seq['shape1']])
        shape_sub_text = seq['template'].format(
            color1=seq['color1'], shape1=new_shape1,
            color2=seq['color2'], shape2=seq['shape2']
        )
        
        edits.append({
            'base_text': base_text,
            'order_swap': swap_text,
            'color_substitution': color_sub_text,
            'shape_substitution': shape_sub_text,
            'base_id': i
        })
    
    return edits

def compute_embeddings(texts, use_bigrams=False):
    """Compute embeddings using HashingVectorizer"""
    if use_bigrams:
        vectorizer = HashingVectorizer(
            n_features=2**16,  # Large dimension to avoid collisions
            ngram_range=(1, 2),  # Unigrams + bigrams
            analyzer='word',
            norm='l2'
        )
    else:
        vectorizer = HashingVectorizer(
            n_features=2**16,  # Large dimension to avoid collisions
            ngram_range=(1, 1),  # Unigrams only
            analyzer='word',
            norm='l2'
        )
    
    embeddings = vectorizer.fit_transform(texts)
    return embeddings.toarray()

def calculate_cosine_distance(vec1, vec2):
    """Calculate cosine distance between two vectors"""
    similarity = cosine_similarity([vec1], [vec2])[0, 0]
    return 1.0 - similarity

def run_experiment():
    """Run the complete experiment"""
    print("Starting embedding order sensitivity experiment...")
    start_time = time.time()
    
    # Generate data
    base_sequences = generate_object_sequences(n_base=120)
    edits = create_edits(base_sequences)
    
    print(f"Generated {len(edits)} edit sets")
    
    # Collect all texts for embedding
    all_texts = []
    for edit_set in edits:
        all_texts.extend([
            edit_set['base_text'],
            edit_set['order_swap'],
            edit_set['color_substitution'],
            edit_set['shape_substitution']
        ])
    
    print("Computing unigram embeddings...")
    unigram_embeddings = compute_embeddings(all_texts, use_bigrams=False)
    
    print("Computing unigram+bigram embeddings...")
    bigram_embeddings = compute_embeddings(all_texts, use_bigrams=True)
    
    # Calculate distances for each condition
    results = []
    
    for i, edit_set in enumerate(edits):
        if i % 25 == 0:
            print(f"Computing distances for edit set {i+1}/{len(edits)}...")
            
        base_idx = i * 4
        swap_idx = base_idx + 1
        color_idx = base_idx + 2
        shape_idx = base_idx + 3
        
        # Unigram distances
        unigram_order_dist = calculate_cosine_distance(
            unigram_embeddings[base_idx], unigram_embeddings[swap_idx]
        )
        unigram_color_dist = calculate_cosine_distance(
            unigram_embeddings[base_idx], unigram_embeddings[color_idx]
        )
        unigram_shape_dist = calculate_cosine_distance(
            unigram_embeddings[base_idx], unigram_embeddings[shape_idx]
        )
        
        # Bigram distances
        bigram_order_dist = calculate_cosine_distance(
            bigram_embeddings[base_idx], bigram_embeddings[swap_idx]
        )
        bigram_color_dist = calculate_cosine_distance(
            bigram_embeddings[base_idx], bigram_embeddings[color_idx]
        )
        bigram_shape_dist = calculate_cosine_distance(
            bigram_embeddings[base_idx], bigram_embeddings[shape_idx]
        )
        
        # Calculate sensitivity ratios
        unigram_attr_dist = (unigram_color_dist + unigram_shape_dist) / 2
        bigram_attr_dist = (bigram_color_dist + bigram_shape_dist) / 2
        
        unigram_ratio = unigram_order_dist / unigram_attr_dist if unigram_attr_dist > 0 else 0.0
        bigram_ratio = bigram_order_dist / bigram_attr_dist if bigram_attr_dist > 0 else 0.0
        
        results.append({
            'edit_id': i,
            'base_text': edit_set['base_text'],
            'order_swap_text': edit_set['order_swap'],
            'unigram_order_distance': float(unigram_order_dist),
            'unigram_color_distance': float(unigram_color_dist),
            'unigram_shape_distance': float(unigram_shape_dist),
            'unigram_avg_attribute_distance': float(unigram_attr_dist),
            'unigram_sensitivity_ratio': float(unigram_ratio),
            'bigram_order_distance': float(bigram_order_dist),
            'bigram_color_distance': float(bigram_color_dist),
            'bigram_shape_distance': float(bigram_shape_dist),
            'bigram_avg_attribute_distance': float(bigram_attr_dist),
            'bigram_sensitivity_ratio': float(bigram_ratio),
            'ratio_difference': float(bigram_ratio - unigram_ratio)
        })
    
    print(f"\nCompleted analysis of {len(results)} data points")
    
    # Verify we have enough data points
    if len(results) < 50:
        raise ValueError(f"Insufficient data points: {len(results)} < 50")
    
    # Calculate summary statistics
    unigram_ratios = [r['unigram_sensitivity_ratio'] for r in results]
    bigram_ratios = [r['bigram_sensitivity_ratio'] for r in results]
    ratio_differences = [r['ratio_difference'] for r in results]
    
    avg_unigram_ratio = np.mean(unigram_ratios)
    avg_bigram_ratio = np.mean(bigram_ratios)
    avg_ratio_diff = np.mean(ratio_differences)
    
    # Check success criteria
    success_threshold_diff = 0.35
    success_threshold_abs = 0.40
    
    meets_diff_threshold = avg_ratio_diff >= success_threshold_diff
    meets_abs_threshold = avg_bigram_ratio >= success_threshold_abs
    experiment_success = meets_diff_threshold and meets_abs_threshold
    
    # Prepare final results
    final_results = {
        'experiment_config': {
            'n_base_sequences': 120,
            'n_data_points': len(results),
            'embedding_dimensions': 2**16,
            'success_threshold_difference': success_threshold_diff,
            'success_threshold_absolute': success_threshold_abs
        },
        'summary_statistics': {
            'avg_unigram_sensitivity_ratio': float(avg_unigram_ratio),
            'avg_bigram_sensitivity_ratio': float(avg_bigram_ratio),
            'avg_ratio_difference': float(avg_ratio_diff),
            'std_unigram_ratio': float(np.std(unigram_ratios)),
            'std_bigram_ratio': float(np.std(bigram_ratios)),
            'std_ratio_difference': float(np.std(ratio_differences))
        },
        'success_criteria': {
            'meets_difference_threshold': meets_diff_threshold,
            'meets_absolute_threshold': meets_abs_threshold,
            'experiment_success': experiment_success
        },
        'detailed_results': results,
        'runtime_seconds': float(time.time() - start_time)
    }
    
    # Save results
    os.makedirs('/Users/kumacmini/cost-aware-research-search/results', exist_ok=True)
    output_path = '/Users/kumacmini/cost-aware-research-search/results/iter_01_results.json'
    
    with open(output_path, 'w') as f:
        json.dump(final_results, f, indent=2, default=str)
    
    print(f"\nResults saved to {output_path}")
    
    # Print final results table
    print("\n" + "="*80)
    print("FINAL EXPERIMENT RESULTS")
    print("="*80)
    print(f"Total data points: {len(results)}")
    print(f"Runtime: {final_results['runtime_seconds']:.2f} seconds")
    print("\nSENSITIVITY RATIO ANALYSIS:")
    print(f"  Unigram baseline ratio:     {avg_unigram_ratio:.4f} ± {np.std(unigram_ratios):.4f}")
    print(f"  Unigram+bigram ratio:       {avg_bigram_ratio:.4f} ± {np.std(bigram_ratios):.4f}")
    print(f"  Difference:                 {avg_ratio_diff:.4f} ± {np.std(ratio_differences):.4f}")
    print("\nSUCCESS CRITERIA:")
    print(f"  Difference ≥ {success_threshold_diff}:        {'✓' if meets_diff_threshold else '✗'} ({avg_ratio_diff:.4f})")
    print(f"  Absolute ratio ≥ {success_threshold_abs}:      {'✓' if meets_abs_threshold else '✗'} ({avg_bigram_ratio:.4f})")
    print(f"  Overall success:             {'✓' if experiment_success else '✗'}")
    print("="*80)
    
    return final_results

if __name__ == "__main__":
    results = run_experiment()