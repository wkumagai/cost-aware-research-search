import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import random
from collections import defaultdict
import statistics

def generate_sentences_by_complexity():
    """Generate sentences with varying syntactic complexity"""
    
    # Simple sentences (complexity level 1)
    simple_sentences = [
        "The cat sits on the mat.",
        "Dogs bark loudly.",
        "She runs fast.",
        "Birds fly high.",
        "Rain falls down.",
        "The sun shines bright.",
        "Fish swim quickly.",
        "Trees grow tall.",
        "Cars move fast.",
        "Wind blows hard.",
        "Stars shine at night.",
        "Children play games.",
        "Books contain stories.",
        "Flowers smell nice.",
        "Water feels cold.",
        "Fire burns hot.",
        "Snow melts slowly.",
        "Grass grows green.",
        "Music sounds beautiful.",
        "Food tastes good."
    ]
    
    # Medium complexity sentences (complexity level 2)
    medium_sentences = [
        "The large brown dog that lives next door barks every morning.",
        "When the sun sets, the sky becomes orange and pink.",
        "Although it was raining, the children continued playing outside.",
        "The book that I read yesterday was incredibly fascinating.",
        "Because she studied hard, she passed the difficult exam.",
        "The old tree in the garden provides shade during summer.",
        "While cooking dinner, she listened to her favorite music.",
        "The students who attended the lecture took detailed notes.",
        "After finishing work, he went to the gym for exercise.",
        "The beautiful flowers in the vase filled the room with fragrance.",
        "Since the weather was nice, they decided to have a picnic.",
        "The movie that we watched last night was surprisingly good.",
        "Before leaving home, she checked the weather forecast carefully.",
        "The cat sleeping on the windowsill looked very peaceful.",
        "During the storm, the sailors struggled to control their boat.",
        "The teacher who explained the concept was very patient.",
        "As the clock struck midnight, the party finally ended.",
        "The artist whose paintings hang in galleries is quite famous.",
        "Unless you practice regularly, you won't improve your skills.",
        "The coffee that she made this morning tasted exceptionally bitter."
    ]
    
    # Complex sentences (complexity level 3)
    complex_sentences = [
        "The research team, which had been working on the project for months, discovered that the hypothesis they had initially proposed was not only incorrect but also fundamentally flawed in its basic assumptions.",
        "Although the committee members disagreed about the budget allocation, they eventually reached a compromise that satisfied most stakeholders, despite the fact that some concerns remained unaddressed.",
        "The philosopher argued that consciousness, being the most mysterious aspect of human experience, cannot be fully explained by neuroscience alone, since it involves subjective phenomena that resist objective measurement.",
        "While the economic indicators suggested a recovery was imminent, the experts warned that several factors, including geopolitical tensions and supply chain disruptions, could potentially derail the anticipated growth.",
        "The novel, which explores themes of identity and belonging through the lens of a immigrant family's multi-generational saga, demonstrates how cultural adaptation involves both preservation and transformation of traditional values.",
        "Because the experimental results contradicted the established theory, the scientists had to reconsider their fundamental assumptions about the phenomenon, which led to a paradigm shift in the field.",
        "The architect's design, incorporating sustainable materials and energy-efficient systems while maintaining aesthetic appeal, represents a synthesis of environmental consciousness and artistic vision.",
        "Since the diplomatic negotiations involved multiple parties with conflicting interests, the mediators employed various strategies to find common ground, though success remained uncertain.",
        "The historian's analysis revealed that the events, previously understood as isolated incidents, were actually interconnected parts of a larger pattern that shaped the entire historical period.",
        "When the technological breakthrough occurred, it not only revolutionized the industry but also raised ethical questions about artificial intelligence that society was not yet prepared to answer.",
        "The psychological study examined how childhood experiences influence adult behavior patterns, particularly focusing on the mechanisms through which early trauma affects emotional regulation and interpersonal relationships.",
        "Although the mathematical proof was elegant in its simplicity, it required sophisticated understanding of abstract concepts that challenged even experienced researchers in the field.",
        "The environmental scientist explained that climate change, being a complex system with multiple feedback loops, produces effects that are often delayed and non-linear, making prediction extremely difficult.",
        "The composer's symphony, written during a period of personal turmoil, reflects both the chaos of emotional upheaval and the search for meaning through artistic expression.",
        "Since the legal precedent had not been clearly established, the judges had to weigh competing interpretations of constitutional principles while considering the broader implications for future cases.",
        "The anthropologist's fieldwork revealed that cultural practices, seemingly arbitrary to outsiders, actually serve important social functions that maintain community cohesion and identity.",
        "While the medical treatment showed promise in clinical trials, the researchers emphasized that long-term effects remained unknown and that patient monitoring would be essential.",
        "The political scientist argued that democratic institutions, though imperfect, provide the best available mechanism for peaceful conflict resolution and social progress.",
        "Because the engineering challenge required interdisciplinary collaboration, teams combining expertise in materials science, computer modeling, and manufacturing processes worked together to develop solutions.",
        "The literary critic's interpretation suggested that the author's use of symbolism, while appearing straightforward on the surface, actually contained multiple layers of meaning that reflected broader cultural anxieties."
    ]
    
    # Create labeled dataset
    sentences = []
    complexity_labels = []
    
    # Use all sentences to ensure we have enough samples
    for sent in simple_sentences:
        sentences.append(sent)
        complexity_labels.append(1)
    
    for sent in medium_sentences:
        sentences.append(sent)
        complexity_labels.append(2)
    
    for sent in complex_sentences:
        sentences.append(sent)
        complexity_labels.append(3)
    
    return sentences, complexity_labels

def calculate_silhouette_for_random_grouping(embeddings, n_clusters=3, n_trials=5):
    """Calculate silhouette score for random groupings"""
    scores = []
    for _ in range(n_trials):
        random_labels = np.random.randint(0, n_clusters, len(embeddings))
        if len(set(random_labels)) > 1:  # Need at least 2 clusters
            score = silhouette_score(embeddings, random_labels)
            scores.append(score)
    return np.mean(scores) if scores else 0

def analyze_embedding_clusters():
    """Main analysis function"""
    print("Starting sentence embedding clustering analysis...")
    print("=" * 60)
    
    # Generate sentences
    print("Generating sentences with varying syntactic complexity...")
    sentences, complexity_labels = generate_sentences_by_complexity()
    print(f"Generated {len(sentences)} sentences:")
    print(f"  - Simple (level 1): {complexity_labels.count(1)} sentences")
    print(f"  - Medium (level 2): {complexity_labels.count(2)} sentences") 
    print(f"  - Complex (level 3): {complexity_labels.count(3)} sentences")
    
    # Load sentence transformer model
    print("\nLoading sentence transformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Generate embeddings
    print("Generating sentence embeddings...")
    embeddings = model.encode(sentences)
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Calculate complexity-based clustering silhouette score
    print("\nCalculating silhouette scores...")
    complexity_silhouette = silhouette_score(embeddings, complexity_labels)
    print(f"Complexity-based clustering silhouette score: {complexity_silhouette:.4f}")
    
    # Calculate random clustering baseline
    random_silhouette = calculate_silhouette_for_random_grouping(embeddings, n_clusters=3)
    print(f"Random grouping silhouette score (baseline): {random_silhouette:.4f}")
    
    # Calculate success metric
    silhouette_improvement = complexity_silhouette - random_silhouette
    success_threshold = 0.1
    success = silhouette_improvement > success_threshold
    
    print(f"\nSuccess Analysis:")
    print(f"  Silhouette improvement: {silhouette_improvement:.4f}")
    print(f"  Success threshold: {success_threshold:.4f}")
    print(f"  Success achieved: {success}")
    
    # Perform K-means clustering for comparison
    print("\nPerforming K-means clustering...")
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(embeddings)
    kmeans_silhouette = silhouette_score(embeddings, kmeans_labels)
    print(f"K-means clustering silhouette score: {kmeans_silhouette:.4f}")
    
    # Analyze cluster characteristics
    print("\nAnalyzing cluster characteristics by complexity level...")
    cluster_stats = defaultdict(list)
    
    for i, complexity in enumerate(complexity_labels):
        embedding = embeddings[i]
        cluster_stats[complexity].append(embedding)
    
    # Calculate within-cluster variance for each complexity level
    complexity_variances = {}
    for complexity, embedding_list in cluster_stats.items():
        if len(embedding_list) > 1:
            embeddings_array = np.array(embedding_list)
            centroid = np.mean(embeddings_array, axis=0)
            distances = [np.linalg.norm(emb - centroid) for emb in embeddings_array]
            variance = np.var(distances)
            complexity_variances[complexity] = {
                'variance': variance,
                'mean_distance': np.mean(distances),
                'std_distance': np.std(distances),
                'count': len(embedding_list)
            }
    
    print("\nCluster variance analysis:")
    for complexity in sorted(complexity_variances.keys()):
        stats = complexity_variances[complexity]
        print(f"  Complexity level {complexity}:")
        print(f"    Count: {stats['count']}")
        print(f"    Distance variance: {stats['variance']:.6f}")
        print(f"    Mean distance to centroid: {stats['mean_distance']:.4f}")
        print(f"    Std distance to centroid: {stats['std_distance']:.4f}")
    
    # Dimensionality reduction for visualization
    print("\nGenerating visualization...")
    pca = PCA(n_components=2, random_state=42)
    embeddings_2d = pca.fit_transform(embeddings)
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    colors = ['red', 'blue', 'green']
    complexity_names = ['Simple', 'Medium', 'Complex']
    
    for i, complexity in enumerate([1, 2, 3]):
        mask = np.array(complexity_labels) == complexity
        plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                   c=colors[i], label=f'{complexity_names[i]} (Level {complexity})', 
                   alpha=0.7, s=50)
    
    plt.title('Sentence Embeddings by Syntactic Complexity\n(PCA Projection)', fontsize=14)
    plt.xlabel(f'First Principal Component (explains {pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'Second Principal Component (explains {pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    results_dir = "/Users/kumacmini/cost-aware-research-search/results"
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(os.path.join(results_dir, "embedding_clusters_visualization.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Prepare results
    results = {
        'experiment_info': {
            'hypothesis': 'Sentence embeddings from transformer models will cluster differently based on syntactic complexity',
            'n_samples': len(sentences),
            'embedding_model': 'all-MiniLM-L6-v2',
            'embedding_dim': embeddings.shape[1],
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        },
        'data_distribution': {
            'simple_sentences': complexity_labels.count(1),
            'medium_sentences': complexity_labels.count(2), 
            'complex_sentences': complexity_labels.count(3)
        },
        'clustering_results': {
            'complexity_based_silhouette': float(complexity_silhouette),
            'random_baseline_silhouette': float(random_silhouette),
            'kmeans_silhouette': float(kmeans_silhouette),
            'silhouette_improvement': float(silhouette_improvement),
            'success_threshold': success_threshold,
            'success_achieved': success
        },
        'cluster_analysis': {
            f'complexity_{k}': {
                'variance': float(v['variance']),
                'mean_distance': float(v['mean_distance']),
                'std_distance': float(v['std_distance']),
                'sample_count': int(v['count'])
            } for k, v in complexity_variances.items()
        },
        'pca_analysis': {
            'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
            'total_variance_explained': float(sum(pca.explained_variance_ratio_))
        }
    }
    
    # Save results
    results_file = os.path.join(results_dir, "iter_01_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    print("Visualization saved to: embedding_clusters_visualization.png")
    
    # Print summary table
    print("\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY")
    print("=" * 60)
    
    print(f"{'Metric':<30} {'Value':<15} {'Status':<10}")
    print("-" * 55)
    print(f"{'Total samples':<30} {len(sentences):<15} {'✓':<10}")
    print(f"{'Complexity silhouette':<30} {complexity_silhouette:<15.4f} {'✓' if complexity_silhouette > 0 else '✗':<10}")
    print(f"{'Random baseline':<30} {random_silhouette:<15.4f} {'-':<10}")
    print(f"{'Improvement':<30} {silhouette_improvement:<15.4f} {'✓' if success else '✗':<10}")
    print(f"{'Success threshold':<30} {success_threshold:<15.4f} {'-':<10}")
    print(f"{'Hypothesis supported':<30} {str(success):<15} {'✓' if success else '✗':<10}")
    
    # Variance analysis summary
    print(f"\nComplexity Level Variance Analysis:")
    print(f"{'Level':<10} {'Count':<8} {'Variance':<12} {'Mean Dist':<12}")
    print("-" * 42)
    for complexity in sorted(complexity_variances.keys()):
        stats = complexity_variances[complexity]
        level_name = ['Simple', 'Medium', 'Complex'][complexity-1]
        print(f"{level_name:<10} {stats['count']:<8} {stats['variance']:<12.6f} {stats['mean_distance']:<12.4f}")
    
    print(f"\nConclusion: {'SUCCESS' if success else 'FAILURE'}")
    if success:
        print("The hypothesis is supported - sentence embeddings cluster differently based on syntactic complexity.")
    else:
        print("The hypothesis is not strongly supported - clustering by complexity shows limited improvement over random grouping.")
    
    return results

if __name__ == "__main__":
    start_time = time.time()
    
    try:
        results = analyze_embedding_clusters()
        
        end_time = time.time()
        runtime = end_time - start_time
        
        print(f"\nExperiment completed successfully in {runtime:.1f} seconds")
        
    except Exception as e:
        print(f"Experiment failed with error: {str(e)}")
        
        # Save error results
        error_results = {
            'experiment_info': {
                'hypothesis': 'Sentence embeddings cluster differently based on syntactic complexity',
                'status': 'failed',
                'error': str(e),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'clustering_results': {
                'success_achieved': False,
                'error_message': str(e)
            }
        }
        
        results_dir = "/Users/kumacmini/cost-aware-research-search/results"
        os.makedirs(results_dir, exist_ok=True)
        results_file = os.path.join(results_dir, "iter_01_results.json")
        
        with open(results_file, 'w') as f:
            json.dump(error_results, f, indent=2)
        
        print(f"Error results saved to: {results_file}")