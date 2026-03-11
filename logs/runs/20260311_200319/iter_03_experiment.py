import json
import os
import sys
import time
import random
import math
from collections import defaultdict
import statistics
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import openai

def setup_api():
    """Setup OpenAI API client"""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    openai.api_key = api_key
    return openai

def generate_paraphrase_with_api(client, original_sentence, style):
    """Generate paraphrase using OpenAI API with retry logic"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that creates paraphrases. Maintain the exact same meaning while changing the syntactic structure."},
                    {"role": "user", "content": f"Paraphrase this sentence in {style} style, keeping identical meaning: '{original_sentence}'"}
                ],
                max_completion_tokens=50,
                temperature=0.7
            )
            return response.choices[0].message.content.strip().strip('"').strip("'")
        except Exception as e:
            print(f"API call attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                return None
    return None

def create_synthetic_dataset():
    """Create balanced synthetic dataset with paraphrases and unrelated sentences"""
    
    # Base sentences covering different semantic domains
    base_sentences = [
        "The cat sat on the mat",
        "Scientists discovered a new planet",
        "The restaurant serves delicious pizza",
        "Students study hard for exams",
        "Rain falls from dark clouds",
        "Dogs bark at strangers",
        "Flowers bloom in spring",
        "Cars drive on highways",
        "Books contain valuable knowledge",
        "Birds fly south in winter",
        "Teachers help students learn",
        "Computers process information quickly",
        "Mountains reach toward the sky",
        "Ocean waves crash on shore",
        "Music brings people joy",
        "Artists create beautiful paintings",
        "Chefs prepare tasty meals",
        "Athletes train every day",
        "Parents love their children",
        "Friends share happy moments"
    ]
    
    # Paraphrase styles for syntactic variation
    paraphrase_styles = ["formal", "casual", "passive voice"]
    
    # Unrelated sentences (different semantic domains)
    unrelated_pool = [
        "The quantum computer solved complex equations",
        "Ancient civilizations built impressive monuments",
        "Photographers capture fleeting moments",
        "Surgeons perform delicate operations",
        "Gardeners plant colorful flowers",
        "Architects design modern buildings",
        "Pilots navigate through stormy weather",
        "Dancers move gracefully across stage",
        "Writers craft compelling stories",
        "Engineers build sturdy bridges",
        "Farmers harvest golden wheat",
        "Lawyers argue important cases",
        "Nurses care for sick patients",
        "Firefighters rescue trapped victims",
        "Sailors navigate vast oceans",
        "Miners extract precious metals",
        "Bakers create delicious pastries",
        "Tailors sew elegant garments",
        "Mechanics repair broken engines",
        "Librarians organize countless books"
    ]
    
    dataset = []
    api_calls_made = 0
    max_api_calls = 25  # Reserve 5 calls for potential retries
    
    client = setup_api()
    
    print("Generating synthetic dataset...")
    
    # Generate paraphrase clusters (semantic equivalents)
    for i, base_sentence in enumerate(base_sentences[:15]):  # Limit to avoid too many API calls
        if api_calls_made >= max_api_calls:
            break
            
        cluster_sentences = [base_sentence]
        
        # Generate 2-3 paraphrases for each base sentence
        for style in paraphrase_styles[:2]:  # Limit styles to control API calls
            if api_calls_made >= max_api_calls:
                break
                
            paraphrase = generate_paraphrase_with_api(client, base_sentence, style)
            api_calls_made += 1
            
            if paraphrase and paraphrase != base_sentence:
                cluster_sentences.append(paraphrase)
            
            time.sleep(0.1)  # Rate limiting
        
        # Add cluster to dataset
        for sentence in cluster_sentences:
            dataset.append({
                'sentence': sentence,
                'cluster_id': f'semantic_{i}',
                'condition': 'semantic_equivalent',
                'position': len(dataset)
            })
    
    # Add unrelated sentences
    random.shuffle(unrelated_pool)
    for i, sentence in enumerate(unrelated_pool[:25]):  # Add enough unrelated sentences
        dataset.append({
            'sentence': sentence,
            'cluster_id': f'unrelated_{i}',
            'condition': 'unrelated',
            'position': len(dataset)
        })
    
    # Shuffle dataset to randomize order effects
    random.shuffle(dataset)
    
    print(f"Generated dataset with {len(dataset)} sentences using {api_calls_made} API calls")
    return dataset

def compute_embeddings(sentences, model_name='all-MiniLM-L6-v2'):
    """Compute sentence embeddings"""
    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    
    print("Computing embeddings...")
    embeddings = model.encode(sentences, show_progress_bar=True)
    return embeddings

def calculate_distance_metrics(dataset, embeddings):
    """Calculate within-cluster and between-cluster distances"""
    
    # Group by semantic clusters
    semantic_clusters = defaultdict(list)
    unrelated_indices = []
    
    for i, item in enumerate(dataset):
        if item['condition'] == 'semantic_equivalent':
            semantic_clusters[item['cluster_id']].append(i)
        else:
            unrelated_indices.append(i)
    
    within_cluster_distances = []
    between_cluster_distances = []
    
    # Calculate within-cluster distances (semantic equivalents)
    for cluster_id, indices in semantic_clusters.items():
        if len(indices) > 1:
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    idx1, idx2 = indices[i], indices[j]
                    # Use cosine distance (1 - cosine similarity)
                    distance = 1 - cosine_similarity([embeddings[idx1]], [embeddings[idx2]])[0][0]
                    within_cluster_distances.append(distance)
    
    # Calculate between-cluster distances (semantic pairs vs unrelated)
    # Compare semantic clusters to unrelated sentences
    all_semantic_indices = []
    for indices in semantic_clusters.values():
        all_semantic_indices.extend(indices)
    
    # Sample to avoid too many comparisons
    semantic_sample = random.sample(all_semantic_indices, min(20, len(all_semantic_indices)))
    unrelated_sample = random.sample(unrelated_indices, min(20, len(unrelated_indices)))
    
    for sem_idx in semantic_sample:
        for unrel_idx in unrelated_sample:
            distance = 1 - cosine_similarity([embeddings[sem_idx]], [embeddings[unrel_idx]])[0][0]
            between_cluster_distances.append(distance)
    
    return within_cluster_distances, between_cluster_distances

def statistical_analysis(within_distances, between_distances):
    """Perform statistical analysis of distance differences"""
    
    within_mean = statistics.mean(within_distances) if within_distances else 0
    within_std = statistics.stdev(within_distances) if len(within_distances) > 1 else 0
    
    between_mean = statistics.mean(between_distances) if between_distances else 0
    between_std = statistics.stdev(between_distances) if len(between_distances) > 1 else 0
    
    # Calculate effect size (difference in means relative to pooled std)
    pooled_std = math.sqrt((within_std**2 + between_std**2) / 2) if within_std > 0 and between_std > 0 else 1
    effect_size = (between_mean - within_mean) / pooled_std if pooled_std > 0 else 0
    
    # Calculate confidence intervals (approximate)
    within_ci_lower = within_mean - 1.96 * (within_std / math.sqrt(len(within_distances))) if len(within_distances) > 0 else 0
    within_ci_upper = within_mean + 1.96 * (within_std / math.sqrt(len(within_distances))) if len(within_distances) > 0 else 0
    
    between_ci_lower = between_mean - 1.96 * (between_std / math.sqrt(len(between_distances))) if len(between_distances) > 0 else 0
    between_ci_upper = between_mean + 1.96 * (between_std / math.sqrt(len(between_distances))) if len(between_distances) > 0 else 0
    
    # Percentage difference
    percent_difference = ((between_mean - within_mean) / between_mean * 100) if between_mean > 0 else 0
    
    # Simple hypothesis test (Welch's t-test approximation)
    if len(within_distances) > 1 and len(between_distances) > 1:
        se_diff = math.sqrt((within_std**2 / len(within_distances)) + (between_std**2 / len(between_distances)))
        t_stat = (between_mean - within_mean) / se_diff if se_diff > 0 else 0
        # Approximate p-value for two-tailed test (rough estimate)
        p_value_approx = 2 * (1 - abs(t_stat) / (abs(t_stat) + math.sqrt(len(within_distances) + len(between_distances))))
    else:
        t_stat = 0
        p_value_approx = 1.0
    
    return {
        'within_cluster_mean': within_mean,
        'within_cluster_std': within_std,
        'within_cluster_ci': [within_ci_lower, within_ci_upper],
        'between_cluster_mean': between_mean,
        'between_cluster_std': between_std,
        'between_cluster_ci': [between_ci_lower, between_ci_upper],
        'effect_size': effect_size,
        'percent_difference': percent_difference,
        't_statistic': t_stat,
        'p_value_approx': p_value_approx,
        'within_sample_size': len(within_distances),
        'between_sample_size': len(between_distances)
    }

def main():
    """Main experimental pipeline"""
    start_time = time.time()
    
    print("=== SEMANTIC INVARIANCE IN SENTENCE EMBEDDINGS EXPERIMENT ===")
    print("Hypothesis: Embeddings from semantically equivalent content cluster closer than unrelated content\n")
    
    try:
        # Generate synthetic dataset
        dataset = create_synthetic_dataset()
        
        if len(dataset) < 50:
            print(f"Warning: Dataset size ({len(dataset)}) is below minimum requirement of 50")
            return
        
        print(f"Dataset created with {len(dataset)} sentences")
        
        # Extract sentences for embedding
        sentences = [item['sentence'] for item in dataset]
        
        # Compute embeddings
        embeddings = compute_embeddings(sentences)
        
        # Calculate distance metrics
        within_distances, between_distances = calculate_distance_metrics(dataset, embeddings)
        
        # Statistical analysis
        stats = statistical_analysis(within_distances, between_distances)
        
        # Check success threshold (20% smaller within-cluster distances)
        success = stats['percent_difference'] >= 20.0
        
        # Print results
        print("\n=== RESULTS ===")
        print(f"Within-cluster distances (semantic equivalents): {stats['within_cluster_mean']:.4f} ± {stats['within_cluster_std']:.4f}")
        print(f"95% CI: [{stats['within_cluster_ci'][0]:.4f}, {stats['within_cluster_ci'][1]:.4f}]")
        print(f"Between-cluster distances (unrelated): {stats['between_cluster_mean']:.4f} ± {stats['between_cluster_std']:.4f}")
        print(f"95% CI: [{stats['between_cluster_ci'][0]:.4f}, {stats['between_cluster_ci'][1]:.4f}]")
        print(f"Effect size: {stats['effect_size']:.3f}")
        print(f"Percentage difference: {stats['percent_difference']:.1f}%")
        print(f"T-statistic: {stats['t_statistic']:.3f}")
        print(f"Approximate p-value: {stats['p_value_approx']:.4f}")
        print(f"Sample sizes: within={stats['within_sample_size']}, between={stats['between_sample_size']}")
        print(f"Success threshold met (≥20% difference): {'YES' if success else 'NO'}")
        
        # Summary table
        print("\n=== SUMMARY TABLE ===")
        print("Metric                    | Within-Cluster | Between-Cluster | Difference")
        print("--------------------------|---------------|----------------|------------")
        print(f"Mean Distance             | {stats['within_cluster_mean']:.4f}       | {stats['between_cluster_mean']:.4f}        | {stats['between_cluster_mean']-stats['within_cluster_mean']:.4f}")
        print(f"Standard Deviation        | {stats['within_cluster_std']:.4f}       | {stats['between_cluster_std']:.4f}        | -")
        print(f"Sample Size               | {stats['within_sample_size']:>13} | {stats['between_sample_size']:>14} | -")
        
        # Prepare results for JSON export
        results = {
            'experiment_name': 'semantic_invariance_embeddings',
            'hypothesis': 'Embeddings from identical semantic content cluster closer than unrelated content',
            'timestamp': time.time(),
            'dataset_size': len(dataset),
            'embedding_model': 'all-MiniLM-L6-v2',
            'statistics': stats,
            'success_threshold_met': success,
            'runtime_seconds': time.time() - start_time,
            'dataset_sample': dataset[:5]  # First 5 items for inspection
        }
        
        # Save results
        output_path = '/Users/kumacmini/cost-aware-research-search/results/iter_03_results.json'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to {output_path}")
        print(f"Total runtime: {time.time() - start_time:.1f} seconds")
        
    except Exception as e:
        print(f"Experiment failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    random.seed(42)  # For reproducibility
    main()