import json
import os
import sys
import time
import math
import random
import numpy as np
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from scipy.spatial.distance import pdist, squareform
from scipy.stats import ttest_ind
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt

def load_embedding_model():
    """Load a sentence transformer model."""
    print("Loading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model

def get_antonym_pairs():
    """Generate a curated list of antonym pairs across different semantic categories."""
    antonym_pairs = [
        # Basic adjectives
        ("hot", "cold"), ("big", "small"), ("fast", "slow"), ("high", "low"),
        ("bright", "dark"), ("hard", "soft"), ("strong", "weak"), ("old", "new"),
        ("good", "bad"), ("right", "wrong"), ("true", "false"), ("rich", "poor"),
        ("thick", "thin"), ("wide", "narrow"), ("deep", "shallow"), ("heavy", "light"),
        
        # Emotions/states
        ("happy", "sad"), ("love", "hate"), ("hope", "despair"), ("calm", "angry"),
        ("brave", "cowardly"), ("kind", "cruel"), ("generous", "selfish"), ("honest", "dishonest"),
        
        # Actions/verbs
        ("give", "take"), ("push", "pull"), ("open", "close"), ("start", "stop"),
        ("build", "destroy"), ("create", "destroy"), ("expand", "contract"), ("rise", "fall"),
        
        # Spatial/temporal
        ("up", "down"), ("left", "right"), ("front", "back"), ("inside", "outside"),
        ("early", "late"), ("first", "last"), ("before", "after"), ("near", "far"),
        
        # Abstract concepts
        ("success", "failure"), ("knowledge", "ignorance"), ("order", "chaos"), ("peace", "war"),
        ("freedom", "slavery"), ("truth", "lie"), ("beauty", "ugliness"), ("virtue", "vice"),
        
        # Physical properties
        ("rough", "smooth"), ("wet", "dry"), ("loud", "quiet"), ("sharp", "dull"),
        ("clean", "dirty"), ("full", "empty"), ("tight", "loose"), ("fresh", "stale"),
        
        # Additional pairs to reach 50+
        ("accept", "reject"), ("agree", "disagree"), ("attack", "defend"), ("advance", "retreat"),
        ("awake", "asleep"), ("victory", "defeat"), ("increase", "decrease"), ("include", "exclude")
    ]
    return antonym_pairs

def get_random_word_pairs(antonym_pairs, n_pairs):
    """Generate random word pairs from the antonym vocabulary."""
    all_words = []
    for pair in antonym_pairs:
        all_words.extend(pair)
    
    random_pairs = []
    for _ in range(n_pairs):
        word1, word2 = random.sample(all_words, 2)
        if word1 != word2:
            random_pairs.append((word1, word2))
    return random_pairs

def calculate_geometric_regularity(embeddings_dict, word_pairs):
    """Calculate geometric regularity metrics for word pairs."""
    results = []
    
    for word1, word2 in word_pairs:
        if word1 in embeddings_dict and word2 in embeddings_dict:
            emb1 = embeddings_dict[word1]
            emb2 = embeddings_dict[word2]
            
            # Calculate center point
            center = (emb1 + emb2) / 2
            
            # Calculate distances from center
            dist1_to_center = np.linalg.norm(emb1 - center)
            dist2_to_center = np.linalg.norm(emb2 - center)
            
            # Calculate distance between the two words
            pair_distance = np.linalg.norm(emb1 - emb2)
            
            # Geometric regularity metrics
            distance_symmetry = 1 - abs(dist1_to_center - dist2_to_center) / max(dist1_to_center, dist2_to_center)
            
            # Triangle regularity (how close to equilateral triangle with center)
            expected_distance = pair_distance / 2  # In equilateral triangle
            triangle_regularity = 1 - abs(dist1_to_center - expected_distance) / expected_distance
            
            # Cosine similarity (semantic relatedness)
            cosine_sim = cosine_similarity([emb1], [emb2])[0, 0]
            
            results.append({
                'word_pair': (word1, word2),
                'distance_symmetry': distance_symmetry,
                'triangle_regularity': triangle_regularity,
                'pair_distance': pair_distance,
                'cosine_similarity': cosine_sim,
                'center_dist1': dist1_to_center,
                'center_dist2': dist2_to_center
            })
    
    return results

def calculate_geometric_regularity_score(results):
    """Calculate overall geometric regularity score."""
    if not results:
        return 0.0
    
    # Combine multiple geometric measures
    symmetry_scores = [r['distance_symmetry'] for r in results]
    triangle_scores = [r['triangle_regularity'] for r in results]
    
    # Weight the scores
    overall_score = np.mean(symmetry_scores) * 0.6 + np.mean(triangle_scores) * 0.4
    return overall_score

def analyze_embedding_geometry():
    """Main analysis function."""
    print("Starting embedding geometry analysis...")
    
    # Load model and get word pairs
    model = load_embedding_model()
    antonym_pairs = get_antonym_pairs()
    print(f"Loaded {len(antonym_pairs)} antonym pairs")
    
    # Generate random pairs for baseline
    random_pairs = get_random_word_pairs(antonym_pairs, len(antonym_pairs))
    print(f"Generated {len(random_pairs)} random pairs for baseline")
    
    # Get all unique words
    all_words = set()
    for pair in antonym_pairs + random_pairs:
        all_words.update(pair)
    all_words = list(all_words)
    
    print(f"Computing embeddings for {len(all_words)} unique words...")
    
    # Compute embeddings
    embeddings = model.encode(all_words, show_progress_bar=True)
    embeddings_dict = {word: emb for word, emb in zip(all_words, embeddings)}
    
    print("Analyzing geometric relationships...")
    
    # Analyze antonym pairs
    antonym_results = calculate_geometric_regularity(embeddings_dict, antonym_pairs)
    antonym_score = calculate_geometric_regularity_score(antonym_results)
    
    # Analyze random pairs
    random_results = calculate_geometric_regularity(embeddings_dict, random_pairs)
    random_score = calculate_geometric_regularity_score(random_results)
    
    # Statistical comparison
    antonym_symmetries = [r['distance_symmetry'] for r in antonym_results]
    random_symmetries = [r['distance_symmetry'] for r in random_results]
    
    antonym_triangles = [r['triangle_regularity'] for r in antonym_results]
    random_triangles = [r['triangle_regularity'] for r in random_results]
    
    # T-tests
    symmetry_ttest = ttest_ind(antonym_symmetries, random_symmetries)
    triangle_ttest = ttest_ind(antonym_triangles, random_triangles)
    
    # Calculate improvement percentage
    improvement = ((antonym_score - random_score) / random_score) * 100 if random_score > 0 else 0
    
    # Prepare detailed results
    results = {
        'experiment_info': {
            'n_antonym_pairs': len(antonym_pairs),
            'n_random_pairs': len(random_pairs),
            'n_unique_words': len(all_words),
            'embedding_model': 'all-MiniLM-L6-v2',
            'embedding_dimension': embeddings.shape[1]
        },
        'geometric_scores': {
            'antonym_regularity_score': float(antonym_score),
            'random_regularity_score': float(random_score),
            'improvement_percentage': float(improvement),
            'success_threshold_met': improvement >= 20.0
        },
        'detailed_metrics': {
            'antonym_distance_symmetry': {
                'mean': float(np.mean(antonym_symmetries)),
                'std': float(np.std(antonym_symmetries)),
                'median': float(np.median(antonym_symmetries))
            },
            'random_distance_symmetry': {
                'mean': float(np.mean(random_symmetries)),
                'std': float(np.std(random_symmetries)),
                'median': float(np.median(random_symmetries))
            },
            'antonym_triangle_regularity': {
                'mean': float(np.mean(antonym_triangles)),
                'std': float(np.std(antonym_triangles)),
                'median': float(np.median(antonym_triangles))
            },
            'random_triangle_regularity': {
                'mean': float(np.mean(random_triangles)),
                'std': float(np.std(random_triangles)),
                'median': float(np.median(random_triangles))
            }
        },
        'statistical_tests': {
            'symmetry_ttest': {
                'statistic': float(symmetry_ttest.statistic),
                'pvalue': float(symmetry_ttest.pvalue),
                'significant': symmetry_ttest.pvalue < 0.05
            },
            'triangle_ttest': {
                'statistic': float(triangle_ttest.statistic),
                'pvalue': float(triangle_ttest.pvalue),
                'significant': triangle_ttest.pvalue < 0.05
            }
        },
        'sample_results': {
            'top_regular_antonyms': sorted(antonym_results, 
                                         key=lambda x: x['distance_symmetry'], 
                                         reverse=True)[:10],
            'bottom_regular_antonyms': sorted(antonym_results, 
                                            key=lambda x: x['distance_symmetry'])[:10]
        }
    }
    
    return results

def create_visualizations(results):
    """Create visualization plots."""
    print("Creating visualizations...")
    
    # Extract data for plotting
    antonym_sym = [r['distance_symmetry'] for r in results['sample_results']['top_regular_antonyms'] + 
                   results['sample_results']['bottom_regular_antonyms']]
    
    # Create simple histogram comparison
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.hist([results['detailed_metrics']['antonym_distance_symmetry']['mean']], 
             bins=20, alpha=0.7, label='Antonyms', color='blue')
    plt.hist([results['detailed_metrics']['random_distance_symmetry']['mean']], 
             bins=20, alpha=0.7, label='Random', color='red')
    plt.xlabel('Distance Symmetry Score')
    plt.ylabel('Frequency')
    plt.title('Geometric Regularity: Antonyms vs Random')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    categories = ['Antonym Pairs', 'Random Pairs']
    scores = [results['geometric_scores']['antonym_regularity_score'],
              results['geometric_scores']['random_regularity_score']]
    plt.bar(categories, scores, color=['blue', 'red'], alpha=0.7)
    plt.ylabel('Geometric Regularity Score')
    plt.title('Overall Geometric Regularity Comparison')
    
    plt.tight_layout()
    plt.savefig('/Users/kumacmini/cost-aware-research-search/results/iter_01_geometry_plot.png', 
                dpi=150, bbox_inches='tight')
    plt.close()

def print_results_table(results):
    """Print formatted results table."""
    print("\n" + "="*80)
    print("EMBEDDING GEOMETRY ANALYSIS RESULTS")
    print("="*80)
    
    print(f"Dataset Size: {results['experiment_info']['n_antonym_pairs']} antonym pairs")
    print(f"Baseline Size: {results['experiment_info']['n_random_pairs']} random pairs")
    print(f"Embedding Model: {results['experiment_info']['embedding_model']}")
    print(f"Embedding Dimension: {results['experiment_info']['embedding_dimension']}")
    
    print("\nGEOMETRIC REGULARITY SCORES:")
    print("-" * 40)
    print(f"Antonym Pairs Score: {results['geometric_scores']['antonym_regularity_score']:.4f}")
    print(f"Random Pairs Score:  {results['geometric_scores']['random_regularity_score']:.4f}")
    print(f"Improvement:         {results['geometric_scores']['improvement_percentage']:.2f}%")
    print(f"Success Threshold:   {'✓ MET' if results['geometric_scores']['success_threshold_met'] else '✗ NOT MET'} (≥20%)")
    
    print("\nDETAILED METRICS:")
    print("-" * 40)
    print(f"Antonym Distance Symmetry: {results['detailed_metrics']['antonym_distance_symmetry']['mean']:.4f} ± {results['detailed_metrics']['antonym_distance_symmetry']['std']:.4f}")
    print(f"Random Distance Symmetry:  {results['detailed_metrics']['random_distance_symmetry']['mean']:.4f} ± {results['detailed_metrics']['random_distance_symmetry']['std']:.4f}")
    print(f"Antonym Triangle Regularity: {results['detailed_metrics']['antonym_triangle_regularity']['mean']:.4f} ± {results['detailed_metrics']['antonym_triangle_regularity']['std']:.4f}")
    print(f"Random Triangle Regularity:  {results['detailed_metrics']['random_triangle_regularity']['mean']:.4f} ± {results['detailed_metrics']['random_triangle_regularity']['std']:.4f}")
    
    print("\nSTATISTICAL SIGNIFICANCE:")
    print("-" * 40)
    print(f"Symmetry Test p-value: {results['statistical_tests']['symmetry_ttest']['pvalue']:.6f} {'(significant)' if results['statistical_tests']['symmetry_ttest']['significant'] else '(not significant)'}")
    print(f"Triangle Test p-value: {results['statistical_tests']['triangle_ttest']['pvalue']:.6f} {'(significant)' if results['statistical_tests']['triangle_ttest']['significant'] else '(not significant)'}")
    
    print("\nTOP GEOMETRICALLY REGULAR ANTONYM PAIRS:")
    print("-" * 40)
    for i, result in enumerate(results['sample_results']['top_regular_antonyms'][:5]):
        word1, word2 = result['word_pair']
        score = result['distance_symmetry']
        print(f"{i+1}. {word1} ↔ {word2}: {score:.4f}")

def main():
    """Main execution function."""
    start_time = time.time()
    
    try:
        # Run the main analysis
        results = analyze_embedding_geometry()
        
        # Create visualizations
        create_visualizations(results)
        
        # Print results table
        print_results_table(results)
        
        # Add execution metadata
        results['execution_metadata'] = {
            'execution_time_seconds': time.time() - start_time,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'success': True
        }
        
        # Save results to JSON
        output_path = '/Users/kumacmini/cost-aware-research-search/results/iter_01_results.json'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\nResults saved to: {output_path}")
        print(f"Execution time: {time.time() - start_time:.2f} seconds")
        
        # Final summary
        if results['geometric_scores']['success_threshold_met']:
            print("\n🎉 HYPOTHESIS SUPPORTED: Antonym pairs show significantly higher geometric regularity!")
        else:
            print("\n❌ HYPOTHESIS NOT SUPPORTED: No significant geometric regularity difference found.")
            
    except Exception as e:
        print(f"Error during execution: {str(e)}")
        # Save error info
        error_results = {
            'error': str(e),
            'execution_metadata': {
                'execution_time_seconds': time.time() - start_time,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'success': False
            }
        }
        
        output_path = '/Users/kumacmini/cost-aware-research-search/results/iter_01_results.json'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(error_results, f, indent=2)
        
        sys.exit(1)

if __name__ == "__main__":
    main()