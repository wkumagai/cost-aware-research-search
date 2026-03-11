import json
import os
import sys
import time
import random
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from collections import defaultdict
import statistics

def generate_semantic_groups():
    """Generate semantically equivalent sentences in different syntactic forms"""
    semantic_groups = [
        # Group 1: Travel/Journey
        [
            "The cat is sleeping on the warm windowsill in the afternoon sun.",
            "A feline rests peacefully on the heated window ledge during daylight hours.",
            "On the sunny windowsill, a cat dozes contentedly in the warm light."
        ],
        # Group 2: Weather
        [
            "Heavy rain is falling from the dark storm clouds above the city.",
            "Intense precipitation descends from the ominous sky over urban areas.",
            "The metropolitan area experiences substantial rainfall from threatening clouds."
        ],
        # Group 3: Technology
        [
            "The new smartphone features an advanced camera system and long battery life.",
            "This modern mobile device includes sophisticated photography technology and extended power duration.",
            "The latest cellular phone offers enhanced imaging capabilities and prolonged operational time."
        ],
        # Group 4: Food
        [
            "Fresh vegetables and fruits provide essential vitamins and minerals for healthy living.",
            "Newly harvested produce supplies vital nutrients necessary for optimal wellness.",
            "Raw plant foods deliver important nutritional elements required for good health."
        ],
        # Group 5: Education
        [
            "Students learn mathematics through practice problems and theoretical explanations in school.",
            "Pupils acquire mathematical knowledge via exercises and conceptual instruction in educational institutions.",
            "Learners develop numerical skills using problem-solving activities and academic guidance in classrooms."
        ],
        # Group 6: Transportation
        [
            "The red sports car accelerated quickly down the empty highway at sunset.",
            "A crimson racing vehicle gained speed rapidly on the vacant freeway during evening hours.",
            "The scarlet automobile moved swiftly along the deserted road as the sun set."
        ],
        # Group 7: Nature
        [
            "Tall trees sway gently in the cool mountain breeze near the crystal lake.",
            "Lofty forest giants move softly in the refreshing alpine wind beside the clear water.",
            "Elevated woodland specimens bend peacefully in the crisp highland air adjacent to the pristine pond."
        ],
        # Group 8: Work
        [
            "Office employees collaborate on important projects using digital tools and meetings.",
            "Workplace staff cooperate on significant assignments through electronic resources and conferences.",
            "Corporate personnel work together on crucial tasks via technological instruments and group discussions."
        ],
        # Group 9: Entertainment
        [
            "Children enjoy playing creative games in the neighborhood park every weekend.",
            "Young people delight in imaginative activities at the local recreational area each Saturday and Sunday.",
            "Kids find pleasure in inventive play within the community green space during weekend days."
        ],
        # Group 10: Health
        [
            "Regular exercise and balanced nutrition contribute to improved physical fitness and mental wellbeing.",
            "Consistent physical activity and proper dietary habits enhance bodily strength and psychological health.",
            "Routine workout sessions and nutritional balance promote better physical condition and emotional stability."
        ],
        # Group 11: Communication
        [
            "Friends send text messages and emails to stay connected across long distances.",
            "Companions exchange digital correspondence and electronic mail to maintain contact over great spans.",
            "Associates share written communications and internet messages to preserve relationships despite geographical separation."
        ],
        # Group 12: Art
        [
            "The talented artist painted beautiful landscapes using vibrant colors and careful brushstrokes.",
            "A skilled creator produced stunning natural scenes with brilliant hues and precise artistic techniques.",
            "The gifted painter crafted gorgeous countryside views through vivid pigments and meticulous application methods."
        ],
        # Group 13: Science
        [
            "Research scientists conduct experiments to discover new knowledge about natural phenomena.",
            "Academic investigators perform studies to uncover fresh insights regarding environmental processes.",
            "Scholarly researchers execute tests to reveal novel understanding concerning biological and physical systems."
        ],
        # Group 14: Shopping
        [
            "Customers compare prices and quality before making purchasing decisions at retail stores.",
            "Shoppers evaluate costs and standards prior to buying choices in commercial establishments.",
            "Consumers assess expenses and attributes before acquisition selections at marketplace venues."
        ],
        # Group 15: Music
        [
            "Musicians practice scales and compositions daily to improve their instrumental performance skills.",
            "Musical performers rehearse exercises and pieces regularly to enhance their playing abilities.",
            "Artists work on technical patterns and musical works consistently to develop their instrumental proficiency."
        ],
        # Group 16: Sports
        [
            "Athletes train intensively for months to prepare for important competitive tournaments.",
            "Sports competitors exercise rigorously over extended periods to ready themselves for significant championship events.",
            "Professional players condition themselves thoroughly across many weeks to get prepared for major contest competitions."
        ],
        # Group 17: Home
        [
            "The cozy living room features comfortable furniture and warm lighting for family gatherings.",
            "A welcoming family space includes pleasant seating and soft illumination for household meetings.",
            "The inviting lounge area contains relaxing furnishings and gentle brightness for domestic assemblies."
        ]
    ]
    
    # Ensure we have enough groups for the required sample size
    return semantic_groups

def create_experimental_conditions():
    """Create three experimental conditions with balanced semantic groups"""
    semantic_groups = generate_semantic_groups()
    
    conditions = {
        'within_semantic': [],      # Same semantic content, different syntax
        'between_semantic': [],     # Different semantic content
        'mixed_control': []         # Mixed pairs for control
    }
    
    # Condition 1: Within-semantic pairs (same meaning, different syntax)
    for group in semantic_groups:
        if len(group) >= 2:
            # Create pairs within each semantic group
            for i in range(len(group)):
                for j in range(i+1, len(group)):
                    conditions['within_semantic'].append((group[i], group[j], f"group_{semantic_groups.index(group)}"))
    
    # Condition 2: Between-semantic pairs (different meanings)
    for i in range(len(semantic_groups)):
        for j in range(i+1, min(i+5, len(semantic_groups))):  # Limit to avoid too many pairs
            if semantic_groups[i] and semantic_groups[j]:
                conditions['between_semantic'].append((
                    semantic_groups[i][0], 
                    semantic_groups[j][0], 
                    f"cross_group_{i}_{j}"
                ))
    
    # Condition 3: Mixed control (random sampling)
    all_sentences = [sent for group in semantic_groups for sent in group]
    random.shuffle(all_sentences)
    for i in range(0, min(50, len(all_sentences)-1), 2):
        conditions['mixed_control'].append((
            all_sentences[i], 
            all_sentences[i+1], 
            f"mixed_{i//2}"
        ))
    
    return conditions

def compute_embeddings_and_distances(sentence_pairs, model):
    """Compute embeddings and distances for sentence pairs"""
    results = []
    
    print(f"Computing embeddings for {len(sentence_pairs)} sentence pairs...")
    
    for i, (sent1, sent2, pair_id) in enumerate(sentence_pairs):
        try:
            # Generate embeddings
            embeddings = model.encode([sent1, sent2])
            emb1, emb2 = embeddings[0], embeddings[1]
            
            # Compute distances
            cosine_dist = 1 - cosine_similarity([emb1], [emb2])[0][0]
            euclidean_dist = euclidean_distances([emb1], [emb2])[0][0]
            
            results.append({
                'pair_id': pair_id,
                'sentence1': sent1,
                'sentence2': sent2,
                'cosine_distance': float(cosine_dist),
                'euclidean_distance': float(euclidean_dist),
                'embedding1_norm': float(np.linalg.norm(emb1)),
                'embedding2_norm': float(np.linalg.norm(emb2))
            })
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(sentence_pairs)} pairs")
                
        except Exception as e:
            print(f"Error processing pair {i}: {e}")
            continue
    
    return results

def analyze_clustering_performance(results_by_condition):
    """Analyze clustering performance across conditions"""
    analysis = {}
    
    for condition_name, condition_results in results_by_condition.items():
        if not condition_results:
            continue
            
        cosine_distances = [r['cosine_distance'] for r in condition_results]
        euclidean_distances = [r['euclidean_distance'] for r in condition_results]
        
        analysis[condition_name] = {
            'sample_size': len(condition_results),
            'cosine_distance_stats': {
                'mean': statistics.mean(cosine_distances),
                'median': statistics.median(cosine_distances),
                'stdev': statistics.stdev(cosine_distances) if len(cosine_distances) > 1 else 0,
                'min': min(cosine_distances),
                'max': max(cosine_distances)
            },
            'euclidean_distance_stats': {
                'mean': statistics.mean(euclidean_distances),
                'median': statistics.median(euclidean_distances),
                'stdev': statistics.stdev(euclidean_distances) if len(euclidean_distances) > 1 else 0,
                'min': min(euclidean_distances),
                'max': max(euclidean_distances)
            }
        }
    
    return analysis

def perform_hypothesis_test(within_distances, between_distances):
    """Perform simple hypothesis test comparing distance distributions"""
    if not within_distances or not between_distances:
        return {'error': 'Insufficient data for hypothesis test'}
    
    within_mean = statistics.mean(within_distances)
    between_mean = statistics.mean(between_distances)
    
    # Simple effect size calculation
    pooled_std = statistics.stdev(within_distances + between_distances)
    effect_size = abs(between_mean - within_mean) / pooled_std if pooled_std > 0 else 0
    
    # Check success threshold (within-cluster distances should be smaller)
    success_threshold = 0.20  # 20% smaller
    relative_difference = (between_mean - within_mean) / between_mean if between_mean > 0 else 0
    meets_threshold = relative_difference >= success_threshold
    
    return {
        'within_cluster_mean': within_mean,
        'between_cluster_mean': between_mean,
        'relative_difference': relative_difference,
        'effect_size': effect_size,
        'meets_success_threshold': meets_threshold,
        'success_threshold': success_threshold,
        'interpretation': 'SUCCESS' if meets_threshold else 'FAILURE'
    }

def create_visualizations(results_by_condition, output_dir):
    """Create visualizations of the results"""
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Distance distributions by condition
    plt.subplot(1, 3, 1)
    for condition_name, condition_results in results_by_condition.items():
        if condition_results:
            distances = [r['cosine_distance'] for r in condition_results]
            plt.hist(distances, alpha=0.6, label=condition_name, bins=20)
    
    plt.xlabel('Cosine Distance')
    plt.ylabel('Frequency')
    plt.title('Distance Distributions by Condition')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Box plot comparison
    plt.subplot(1, 3, 2)
    box_data = []
    box_labels = []
    for condition_name, condition_results in results_by_condition.items():
        if condition_results:
            distances = [r['cosine_distance'] for r in condition_results]
            box_data.append(distances)
            box_labels.append(condition_name)
    
    if box_data:
        plt.boxplot(box_data, labels=box_labels)
        plt.ylabel('Cosine Distance')
        plt.title('Distance Distribution Comparison')
        plt.xticks(rotation=45)
    
    # Plot 3: Mean distances with error bars
    plt.subplot(1, 3, 3)
    means = []
    stds = []
    labels = []
    
    for condition_name, condition_results in results_by_condition.items():
        if condition_results:
            distances = [r['cosine_distance'] for r in condition_results]
            means.append(statistics.mean(distances))
            stds.append(statistics.stdev(distances) if len(distances) > 1 else 0)
            labels.append(condition_name)
    
    if means:
        plt.bar(labels, means, yerr=stds, capsize=5, alpha=0.7)
        plt.ylabel('Mean Cosine Distance')
        plt.title('Mean Distances by Condition')
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'embedding_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    print("=== Semantic Embedding Clustering Analysis ===")
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Setup
    output_file = "/Users/kumacmini/cost-aware-research-search/results/iter_03_results.json"
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)
    
    # Load embedding model
    print("Loading sentence transformer model...")
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Generate experimental conditions
    print("Generating experimental conditions...")
    conditions = create_experimental_conditions()
    
    print("Experimental Design Summary:")
    for condition_name, pairs in conditions.items():
        print(f"  {condition_name}: {len(pairs)} pairs")
    
    # Ensure minimum sample size
    total_samples = sum(len(pairs) for pairs in conditions.values())
    print(f"Total samples: {total_samples}")
    
    if total_samples < 50:
        print("Warning: Total samples below minimum requirement of 50")
    
    # Process each condition
    results_by_condition = {}
    start_time = time.time()
    
    for condition_name, sentence_pairs in conditions.items():
        print(f"\nProcessing condition: {condition_name}")
        
        # Limit pairs to manage computation time
        limited_pairs = sentence_pairs[:20] if len(sentence_pairs) > 20 else sentence_pairs
        
        condition_results = compute_embeddings_and_distances(limited_pairs, model)
        results_by_condition[condition_name] = condition_results
        
        print(f"Completed {condition_name}: {len(condition_results)} successful computations")
    
    # Analyze results
    print("\n=== ANALYSIS ===")
    analysis = analyze_clustering_performance(results_by_condition)
    
    # Perform hypothesis test
    within_distances = [r['cosine_distance'] for r in results_by_condition.get('within_semantic', [])]
    between_distances = [r['cosine_distance'] for r in results_by_condition.get('between_semantic', [])]
    
    hypothesis_test = perform_hypothesis_test(within_distances, between_distances)
    
    # Create visualizations
    print("Creating visualizations...")
    create_visualizations(results_by_condition, output_dir)
    
    # Compile final results
    final_results = {
        'experiment_info': {
            'hypothesis': 'Embedding vectors from identical semantic content will cluster closer together than embeddings from semantically unrelated content',
            'model_used': 'all-MiniLM-L6-v2',
            'total_runtime_seconds': time.time() - start_time,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        },
        'experimental_conditions': {
            condition: len(pairs) for condition, pairs in conditions.items()
        },
        'analysis': analysis,
        'hypothesis_test': hypothesis_test,
        'detailed_results': results_by_condition
    }
    
    # Save results
    print(f"Saving results to {output_file}")
    with open(output_file, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    # Print summary table
    print("\n=== RESULTS SUMMARY ===")
    print(f"{'Condition':<20} {'N':<8} {'Mean Dist':<12} {'Std Dev':<12}")
    print("-" * 52)
    
    for condition_name, condition_analysis in analysis.items():
        mean_dist = condition_analysis['cosine_distance_stats']['mean']
        std_dist = condition_analysis['cosine_distance_stats']['stdev']
        n_samples = condition_analysis['sample_size']
        print(f"{condition_name:<20} {n_samples:<8} {mean_dist:<12.4f} {std_dist:<12.4f}")
    
    print(f"\nHypothesis Test Results:")
    print(f"Within-cluster mean distance: {hypothesis_test.get('within_cluster_mean', 'N/A'):.4f}")
    print(f"Between-cluster mean distance: {hypothesis_test.get('between_cluster_mean', 'N/A'):.4f}")
    print(f"Relative difference: {hypothesis_test.get('relative_difference', 0):.1%}")
    print(f"Success threshold (20%): {'PASSED' if hypothesis_test.get('meets_success_threshold', False) else 'FAILED'}")
    print(f"Overall result: {hypothesis_test.get('interpretation', 'UNKNOWN')}")
    
    print(f"\nExperiment completed in {time.time() - start_time:.1f} seconds")
    print(f"Results saved to: {output_file}")

if __name__ == "__main__":
    main()