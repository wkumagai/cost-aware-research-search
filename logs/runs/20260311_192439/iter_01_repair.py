import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import textstat
import random
import time
from typing import List, Dict, Tuple

def calculate_syntactic_complexity(sentence: str) -> Dict[str, float]:
    """Calculate multiple syntactic complexity metrics for a sentence."""
    return {
        'flesch_reading_ease': textstat.flesch_reading_ease(sentence),
        'flesch_kincaid_grade': textstat.flesch_kincaid_grade(sentence),
        'automated_readability_index': textstat.automated_readability_index(sentence),
        'coleman_liau_index': textstat.coleman_liau_index(sentence),
        'gunning_fog': textstat.gunning_fog(sentence),
        'smog_index': textstat.smog_index(sentence),
        'lexicon_count': textstat.lexicon_count(sentence),
        'sentence_count': textstat.sentence_count(sentence),
        'syllable_count': textstat.syllable_count(sentence),
        'avg_sentence_length': textstat.avg_sentence_length(sentence),
        'difficult_words': textstat.difficult_words(sentence),
        'linsear_write_formula': textstat.linsear_write_formula(sentence)
    }

def generate_simple_sentences(n: int) -> List[str]:
    """Generate simple sentences with basic structure."""
    subjects = ["The cat", "A dog", "The bird", "My friend", "The teacher", "The student", "A child", "The farmer", "The doctor", "The artist"]
    verbs = ["runs", "walks", "sleeps", "eats", "plays", "works", "reads", "writes", "sings", "dances"]
    objects = ["quickly", "slowly", "peacefully", "happily", "quietly", "loudly", "carefully", "eagerly", "patiently", "gracefully"]
    
    sentences = []
    for _ in range(n):
        subject = random.choice(subjects)
        verb = random.choice(verbs)
        obj = random.choice(objects)
        sentence = f"{subject} {verb} {obj}."
        sentences.append(sentence)
    
    return sentences

def generate_medium_sentences(n: int) -> List[str]:
    """Generate medium complexity sentences with compound structures."""
    simple_clauses = [
        "the weather is nice", "birds are singing", "children are playing", 
        "the sun is shining", "flowers are blooming", "people are walking",
        "cars are driving", "dogs are barking", "music is playing",
        "students are studying", "workers are building", "painters are creating"
    ]
    
    connectors = ["and", "but", "so", "or", "yet"]
    
    sentences = []
    for _ in range(n):
        clause1 = random.choice(simple_clauses)
        clause2 = random.choice(simple_clauses)
        connector = random.choice(connectors)
        
        # Ensure clauses are different
        while clause2 == clause1:
            clause2 = random.choice(simple_clauses)
            
        sentence = f"Today {clause1}, {connector} {clause2}."
        sentences.append(sentence.capitalize())
    
    return sentences

def generate_complex_sentences(n: int) -> List[str]:
    """Generate complex sentences with subordinate clauses and advanced structures."""
    main_clauses = [
        "the research team discovered", "scientists have proven", "the committee decided",
        "experts believe", "the analysis revealed", "investigators found",
        "researchers concluded", "the study demonstrated", "analysts determined",
        "the examination showed", "professionals established", "the investigation uncovered"
    ]
    
    subordinate_starters = [
        "although", "because", "since", "while", "whereas", "despite the fact that",
        "even though", "given that", "considering that", "notwithstanding that"
    ]
    
    subordinate_clauses = [
        "the data collection process was challenging and time-consuming",
        "multiple variables needed to be carefully controlled and monitored",
        "the theoretical framework required extensive modification and refinement",
        "preliminary results indicated unexpected patterns and correlations",
        "the methodology involved sophisticated statistical analysis techniques",
        "external factors significantly influenced the experimental outcomes",
        "the literature review encompassed numerous interdisciplinary perspectives",
        "technological limitations constrained the scope of data analysis",
        "ethical considerations demanded rigorous approval procedures",
        "the sample size requirements exceeded initial projections"
    ]
    
    main_endings = [
        "that the hypothesis warranted further investigation through longitudinal studies",
        "that the theoretical model required substantial conceptual modifications",
        "that the methodology should be replicated across diverse populations",
        "that the findings have significant implications for future research directions",
        "that the results challenge conventional understanding of the phenomenon",
        "that additional validation studies are necessary before definitive conclusions",
        "that the observed patterns reflect underlying systematic relationships",
        "that the experimental design successfully addressed key methodological concerns"
    ]
    
    sentences = []
    for _ in range(n):
        main = random.choice(main_clauses)
        starter = random.choice(subordinate_starters)
        sub_clause = random.choice(subordinate_clauses)
        ending = random.choice(main_endings)
        
        sentence = f"{starter.capitalize()} {sub_clause}, {main} {ending}."
        sentences.append(sentence)
    
    return sentences

def categorize_complexity(complexity_metrics: Dict[str, float]) -> str:
    """Categorize sentence complexity based on multiple metrics."""
    # Use Flesch Reading Ease as primary metric (higher = simpler)
    fre = complexity_metrics['flesch_reading_ease']
    avg_len = complexity_metrics['avg_sentence_length']
    difficult_words = complexity_metrics['difficult_words']
    
    # Combine multiple indicators
    if fre > 60 and avg_len < 15 and difficult_words < 3:
        return 'simple'
    elif fre > 30 and avg_len < 25 and difficult_words < 8:
        return 'medium'
    else:
        return 'complex'

def main():
    print("Starting Sentence Embedding Clustering Analysis")
    start_time = time.time()
    
    results = {
        'experiment_info': {
            'hypothesis': "Sentence embeddings will exhibit distinct clustering patterns based on syntactic complexity",
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'model_used': 'all-MiniLM-L6-v2'
        },
        'data_generation': {},
        'complexity_analysis': {},
        'embedding_analysis': {},
        'clustering_results': {},
        'evaluation_metrics': {},
        'summary': {}
    }
    
    try:
        # Generate sentences with varying complexity
        print("Generating sentences...")
        n_per_category = 34  # Total will be 102 sentences (> 50 minimum)
        simple_sentences = generate_simple_sentences(n_per_category)
        medium_sentences = generate_medium_sentences(n_per_category)
        complex_sentences = generate_complex_sentences(n_per_category)
        
        all_sentences = simple_sentences + medium_sentences + complex_sentences
        expected_labels = (['simple'] * n_per_category + 
                          ['medium'] * n_per_category + 
                          ['complex'] * n_per_category)
        
        print(f"Generated {len(all_sentences)} sentences total")
        
        results['data_generation'] = {
            'total_sentences': len(all_sentences),
            'simple_count': len(simple_sentences),
            'medium_count': len(medium_sentences),
            'complex_count': len(complex_sentences),
            'sample_sentences': {
                'simple': simple_sentences[:3],
                'medium': medium_sentences[:3],
                'complex': complex_sentences[:3]
            }
        }
        
        # Calculate complexity metrics
        print("Calculating complexity metrics...")
        complexity_data = []
        actual_complexity_labels = []
        
        for sentence in all_sentences:
            complexity_metrics = calculate_syntactic_complexity(sentence)
            complexity_data.append(complexity_metrics)
            actual_label = categorize_complexity(complexity_metrics)
            actual_complexity_labels.append(actual_label)
        
        # Analyze complexity distribution
        complexity_counts = {
            'simple': actual_complexity_labels.count('simple'),
            'medium': actual_complexity_labels.count('medium'),
            'complex': actual_complexity_labels.count('complex')
        }
        
        results['complexity_analysis'] = {
            'complexity_distribution': complexity_counts,
            'average_metrics': {
                'simple': {},
                'medium': {},
                'complex': {}
            }
        }
        
        # Calculate average metrics per complexity category
        for category in ['simple', 'medium', 'complex']:
            category_indices = [i for i, label in enumerate(actual_complexity_labels) if label == category]
            if category_indices:
                avg_metrics = {}
                for metric in complexity_data[0].keys():
                    values = [complexity_data[i][metric] for i in category_indices]
                    avg_metrics[metric] = np.mean(values)
                results['complexity_analysis']['average_metrics'][category] = avg_metrics
        
        # Generate sentence embeddings
        print("Generating embeddings...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(all_sentences)
        
        print(f"Generated embeddings with shape: {embeddings.shape}")
        
        results['embedding_analysis'] = {
            'embedding_dimension': embeddings.shape[1],
            'embedding_stats': {
                'mean': float(np.mean(embeddings)),
                'std': float(np.std(embeddings)),
                'min': float(np.min(embeddings)),
                'max': float(np.max(embeddings))
            }
        }
        
        # Perform clustering analysis
        print("Performing clustering analysis...")
        n_clusters = 3
        
        # Cluster all sentences
        kmeans_all = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels_all = kmeans_all.fit_predict(embeddings)
        
        # Calculate silhouette scores
        overall_silhouette = silhouette_score(embeddings, cluster_labels_all)
        
        # Calculate silhouette scores by complexity category
        category_silhouettes = {}
        category_embeddings = {}
        
        for category in ['simple', 'medium', 'complex']:
            category_indices = [i for i, label in enumerate(actual_complexity_labels) if label == category]
            if len(category_indices) > 1:  # Need at least 2 samples for silhouette score
                cat_embeddings = embeddings[category_indices]
                category_embeddings[category] = cat_embeddings
                
                # Cluster within category
                if len(category_indices) >= n_clusters:
                    kmeans_cat = KMeans(n_clusters=min(n_clusters, len(category_indices)), random_state=42, n_init=10)
                    cat_cluster_labels = kmeans_cat.fit_predict(cat_embeddings)
                    
                    if len(np.unique(cat_cluster_labels)) > 1:
                        cat_silhouette = silhouette_score(cat_embeddings, cat_cluster_labels)
                        category_silhouettes[category] = cat_silhouette
                    else:
                        category_silhouettes[category] = None
                else:
                    # Use overall clustering for small categories
                    cat_cluster_labels = cluster_labels_all[category_indices]
                    if len(np.unique(cat_cluster_labels)) > 1:
                        cat_silhouette = silhouette_score(cat_embeddings, cat_cluster_labels)
                        category_silhouettes[category] = cat_silhouette
                    else:
                        category_silhouettes[category] = None
        
        results['clustering_results'] = {
            'n_clusters': n_clusters,
            'overall_silhouette_score': float(overall_silhouette),
            'category_silhouette_scores': {k: float(v) if v is not None else None for k, v in category_silhouettes.items()},
            'cluster_distribution': {
                f'cluster_{i}': int(np.sum(cluster_labels_all == i)) for i in range(n_clusters)
            }
        }
        
        # Generate random baseline
        print("Generating random baseline...")
        random_labels = np.random.randint(0, n_clusters, size=len(all_sentences))
        random_silhouette = silhouette_score(embeddings, random_labels)
        
        results['evaluation_metrics'] = {
            'baseline_random_silhouette': float(random_silhouette),
            'improvement_over_random': float(overall_silhouette - random_silhouette)
        }
        
        # Analyze hypothesis
        simple_score = category_silhouettes.get('simple')
        complex_score = category_silhouettes.get('complex')
        
        hypothesis_supported = False
        score_difference = None
        
        if simple_score is not None and complex_score is not None:
            score_difference = simple_score - complex_score
            # Hypothesis: complex sentences should have 15% lower silhouette score
            threshold = 0.15 * simple_score if simple_score > 0 else 0.15
            hypothesis_supported = score_difference >= threshold
        
        results['summary'] = {
            'hypothesis_supported': hypothesis_supported,
            'score_difference': float(score_difference) if score_difference is not None else None,
            'simple_silhouette': float(simple_score) if simple_score is not None else None,
            'complex_silhouette': float(complex_score) if complex_score is not None else None,
            'runtime_seconds': time.time() - start_time
        }
        
        # Create visualization
        print("Creating visualization...")
        plt.figure(figsize=(12, 8))
        
        # PCA for visualization
        pca = PCA(n_components=2, random_state=42)
        embeddings_2d = pca.fit_transform(embeddings)
        
        # Plot by actual complexity
        plt.subplot(2, 2, 1)
        colors = {'simple': 'blue', 'medium': 'green', 'complex': 'red'}
        for category in ['simple', 'medium', 'complex']:
            indices = [i for i, label in enumerate(actual_complexity_labels) if label == category]
            if indices:
                plt.scatter(embeddings_2d[indices, 0], embeddings_2d[indices, 1], 
                           c=colors[category], label=category, alpha=0.7)
        plt.title('Embeddings by Actual Complexity')
        plt.legend()
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        
        # Plot by cluster assignment
        plt.subplot(2, 2, 2)
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                            c=cluster_labels_all, cmap='viridis', alpha=0.7)
        plt.title('Embeddings by Cluster Assignment')
        plt.colorbar(scatter)
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        
        # Silhouette scores comparison
        plt.subplot(2, 2, 3)
        categories = list(category_silhouettes.keys())
        scores = [category_silhouettes[cat] for cat in categories if category_silhouettes[cat] is not None]
        valid_categories = [cat for cat in categories if category_silhouettes[cat] is not None]
        
        if scores:
            plt.bar(valid_categories, scores, color=['blue', 'green', 'red'][:len(valid_categories)])
            plt.title('Silhouette Scores by Complexity')
            plt.ylabel('Silhouette Score')
            plt.xticks(rotation=45)
        
        # Complexity metrics distribution
        plt.subplot(2, 2, 4)
        flesch_scores = [metrics['flesch_reading_ease'] for metrics in complexity_data]
        plt.hist(flesch_scores, bins=20, alpha=0.7, color='purple')
        plt.title('Distribution of Flesch Reading Ease Scores')
        plt.xlabel('Flesch Reading Ease')
        plt.ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig('/Users/kumacmini/cost-aware-research-search/results/iter_01_embedding_analysis.png', 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        # Print summary
        print("\n" + "="*60)
        print("EXPERIMENT SUMMARY")
        print("="*60)
        print(f"Total sentences analyzed: {len(all_sentences)}")
        print(f"Complexity distribution: {complexity_counts}")
        print(f"Overall silhouette score: {overall_silhouette:.4f}")
        print(f"Random baseline silhouette: {random_silhouette:.4f}")
        print(f"Improvement over random: {overall_silhouette - random_silhouette:.4f}")
        
        print("\nSilhouette scores by complexity:")
        for category, score in category_silhouettes.items():
            if score is not None:
                print(f"  {category}: {score:.4f}")
            else:
                print(f"  {category}: N/A (insufficient data)")
        
        if hypothesis_supported:
            print(f"\n✓ HYPOTHESIS SUPPORTED: Complex sentences show {score_difference:.4f} lower clustering cohesion")
        else:
            print(f"\n✗ HYPOTHESIS NOT SUPPORTED: Difference = {score_difference:.4f}")
        
        print(f"\nRuntime: {time.time() - start_time:.2f} seconds")
        
        # Save results
        output_path = '/Users/kumacmini/cost-aware-research-search/results/iter_01_results.json'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        results['error'] = str(e)
        results['status'] = 'failed'
        
        # Still save partial results
        output_path = '/Users/kumacmini/cost-aware-research-search/results/iter_01_results.json'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        raise

if __name__ == "__main__":
    main()