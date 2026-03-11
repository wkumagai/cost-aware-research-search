import json
import os
import time
import random
import statistics
import numpy as np
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def create_controlled_sentence_dataset():
    """Create semantically controlled sentences with varying syntactic complexity"""
    
    # Base semantic templates with controlled complexity variations
    sentence_templates = [
        # Simple template variations
        {
            'semantic_content': 'cat_sits',
            'simple': "The cat sits.",
            'medium': "The small cat sits quietly.",
            'complex': "The small cat that lives here sits quietly on the chair."
        },
        {
            'semantic_content': 'dog_runs',
            'simple': "The dog runs.",
            'medium': "The brown dog runs quickly.",
            'complex': "The brown dog that barks loudly runs quickly through the park."
        },
        {
            'semantic_content': 'book_interesting',
            'simple': "The book is interesting.",
            'medium': "The old book is very interesting.",
            'complex': "The old book that she recommended is very interesting to read."
        },
        {
            'semantic_content': 'student_studies',
            'simple': "The student studies.",
            'medium': "The dedicated student studies hard.",
            'complex': "The dedicated student who lives nearby studies hard for the exam."
        },
        {
            'semantic_content': 'car_fast',
            'simple': "The car is fast.",
            'medium': "The red car is remarkably fast.",
            'complex': "The red car that he bought is remarkably fast on highways."
        },
        {
            'semantic_content': 'teacher_explains',
            'simple': "The teacher explains.",
            'medium': "The kind teacher explains carefully.",
            'complex': "The kind teacher who knows everything explains carefully to students."
        },
        {
            'semantic_content': 'rain_falls',
            'simple': "Rain falls.",
            'medium': "Heavy rain falls steadily.",
            'complex': "Heavy rain that started yesterday falls steadily on the roof."
        },
        {
            'semantic_content': 'child_plays',
            'simple': "The child plays.",
            'medium': "The happy child plays outside.",
            'complex': "The happy child who loves games plays outside in the garden."
        },
        {
            'semantic_content': 'music_beautiful',
            'simple': "The music is beautiful.",
            'medium': "The classical music is truly beautiful.",
            'complex': "The classical music that she composed is truly beautiful to hear."
        },
        {
            'semantic_content': 'flower_blooms',
            'simple': "The flower blooms.",
            'medium': "The yellow flower blooms brightly.",
            'complex': "The yellow flower that grows here blooms brightly in spring."
        },
        {
            'semantic_content': 'bird_sings',
            'simple': "The bird sings.",
            'medium': "The small bird sings sweetly.",
            'complex': "The small bird that nests nearby sings sweetly every morning."
        },
        {
            'semantic_content': 'water_cold',
            'simple': "The water is cold.",
            'medium': "The lake water is surprisingly cold.",
            'complex': "The lake water that flows from mountains is surprisingly cold in summer."
        },
        {
            'semantic_content': 'man_walks',
            'simple': "The man walks.",
            'medium': "The tall man walks slowly.",
            'complex': "The tall man who wears glasses walks slowly down the street."
        },
        {
            'semantic_content': 'food_delicious',
            'simple': "The food is delicious.",
            'medium': "The homemade food is extremely delicious.",
            'complex': "The homemade food that she prepared is extremely delicious to taste."
        },
        {
            'semantic_content': 'sun_shines',
            'simple': "The sun shines.",
            'medium': "The bright sun shines warmly.",
            'complex': "The bright sun that rises early shines warmly through the clouds."
        },
        {
            'semantic_content': 'door_opens',
            'simple': "The door opens.",
            'medium': "The wooden door opens easily.",
            'complex': "The wooden door that leads outside opens easily with the key."
        },
        {
            'semantic_content': 'computer_works',
            'simple': "The computer works.",
            'medium': "The new computer works perfectly.",
            'complex': "The new computer that costs much works perfectly for all tasks."
        },
        {
            'semantic_content': 'tree_tall',
            'simple': "The tree is tall.",
            'medium': "The oak tree is incredibly tall.",
            'complex': "The oak tree that stands alone is incredibly tall in the forest."
        },
        {
            'semantic_content': 'phone_rings',
            'simple': "The phone rings.",
            'medium': "The old phone rings loudly.",
            'complex': "The old phone that hangs there rings loudly in the kitchen."
        },
        {
            'semantic_content': 'wind_blows',
            'simple': "The wind blows.",
            'medium': "The strong wind blows fiercely.",
            'complex': "The strong wind that comes tonight blows fiercely across the field."
        }
    ]
    
    # Create dataset with explicit complexity metrics
    dataset = []
    
    for template in sentence_templates:
        for complexity_level in ['simple', 'medium', 'complex']:
            sentence = template[complexity_level]
            
            # Calculate explicit complexity metrics
            parse_depth = calculate_parse_depth(sentence)
            clause_count = calculate_clause_count(sentence)
            dependency_distance = calculate_avg_dependency_distance(sentence)
            
            dataset.append({
                'sentence': sentence,
                'semantic_content': template['semantic_content'],
                'complexity_level': complexity_level,
                'parse_depth': parse_depth,
                'clause_count': clause_count,
                'dependency_distance': dependency_distance,
                'complexity_score': parse_depth + clause_count + dependency_distance
            })
    
    return dataset

def calculate_parse_depth(sentence):
    """Simple heuristic for parse depth based on nested structures"""
    depth = 1
    nested_indicators = ['that', 'which', 'who', 'where', 'when']
    for indicator in nested_indicators:
        if indicator in sentence.lower():
            depth += 1
    
    # Count nested punctuation
    paren_depth = 0
    max_paren_depth = 0
    for char in sentence:
        if char == '(':
            paren_depth += 1
            max_paren_depth = max(max_paren_depth, paren_depth)
        elif char == ')':
            paren_depth -= 1
    
    return depth + max_paren_depth

def calculate_clause_count(sentence):
    """Count clauses based on conjunctions and relative pronouns"""
    clause_indicators = ['that', 'which', 'who', 'where', 'when', 'and', 'but', 'or', 'because', 'since', 'while']
    count = 1  # Start with main clause
    
    words = sentence.lower().split()
    for word in words:
        if word.rstrip('.,!?') in clause_indicators:
            count += 1
    
    return count

def calculate_avg_dependency_distance(sentence):
    """Simple heuristic for average dependency distance"""
    words = sentence.split()
    total_distance = 0
    dependencies = 0
    
    # Simple heuristic: measure distance between content words and their modifiers
    for i, word in enumerate(words):
        if word.lower() in ['the', 'a', 'an']:
            # Articles depend on next noun
            if i + 1 < len(words):
                total_distance += 1
                dependencies += 1
        elif word.lower() in ['very', 'extremely', 'quite', 'really']:
            # Adverbs depend on next word
            if i + 1 < len(words):
                total_distance += 1
                dependencies += 1
    
    return total_distance / max(dependencies, 1)

def get_embeddings(sentences, model_name):
    """Get embeddings for sentences using specified model"""
    print(f"Getting embeddings with {model_name}...")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(sentences)
    return embeddings

def calculate_variance_metrics(embeddings, complexity_scores):
    """Calculate variance ratio between high and low complexity embeddings"""
    median_complexity = np.median(complexity_scores)
    
    high_complexity_mask = np.array(complexity_scores) >= median_complexity
    low_complexity_mask = np.array(complexity_scores) < median_complexity
    
    high_complexity_embeddings = embeddings[high_complexity_mask]
    low_complexity_embeddings = embeddings[low_complexity_mask]
    
    high_var = np.mean(np.var(high_complexity_embeddings, axis=0))
    low_var = np.mean(np.var(low_complexity_embeddings, axis=0))
    
    variance_ratio = high_var / max(low_var, 1e-8)
    
    return float(variance_ratio), float(high_var), float(low_var)

def calculate_clustering_quality(embeddings, complexity_labels):
    """Calculate silhouette score for complexity-based clustering"""
    if len(set(complexity_labels)) < 2:
        return 0.0
    
    try:
        score = silhouette_score(embeddings, complexity_labels)
        return float(score)
    except:
        return 0.0

def run_permutation_test(embeddings, complexity_labels, n_permutations=100):
    """Run permutation test for clustering quality"""
    true_score = calculate_clustering_quality(embeddings, complexity_labels)
    
    permutation_scores = []
    for _ in range(n_permutations):
        shuffled_labels = complexity_labels.copy()
        random.shuffle(shuffled_labels)
        perm_score = calculate_clustering_quality(embeddings, shuffled_labels)
        permutation_scores.append(perm_score)
    
    p_value = sum(1 for score in permutation_scores if score >= true_score) / n_permutations
    
    return {
        'true_score': float(true_score),
        'permutation_mean': float(np.mean(permutation_scores)),
        'permutation_std': float(np.std(permutation_scores)),
        'p_value': float(p_value),
        'significant': p_value < 0.05
    }

def supervised_complexity_prediction(embeddings, complexity_scores):
    """Test supervised predictability of complexity from embeddings"""
    # Convert to binary classification (high vs low complexity)
    median_complexity = np.median(complexity_scores)
    binary_labels = [1 if score >= median_complexity else 0 for score in complexity_scores]
    
    if len(set(binary_labels)) < 2:
        return {'accuracy': 0.5, 'feature_importance': []}
    
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            embeddings, binary_labels, test_size=0.3, random_state=42
        )
        
        rf = RandomForestClassifier(n_estimators=50, random_state=42)
        rf.fit(X_train, y_train)
        accuracy = rf.score(X_test, y_test)
        
        return {
            'accuracy': float(accuracy),
            'n_train': len(X_train),
            'n_test': len(X_test)
        }
    except:
        return {'accuracy': 0.5, 'n_train': 0, 'n_test': 0}

def centroid_distance_analysis(embeddings, complexity_labels):
    """Analyze distances between complexity level centroids"""
    unique_labels = list(set(complexity_labels))
    if len(unique_labels) < 2:
        return {'mean_distance': 0.0, 'distances': []}
    
    centroids = {}
    for label in unique_labels:
        mask = np.array(complexity_labels) == label
        if np.sum(mask) > 0:
            centroids[label] = np.mean(embeddings[mask], axis=0)
    
    distances = []
    labels_list = list(centroids.keys())
    for i in range(len(labels_list)):
        for j in range(i + 1, len(labels_list)):
            dist = np.linalg.norm(centroids[labels_list[i]] - centroids[labels_list[j]])
            distances.append(float(dist))
    
    return {
        'mean_distance': float(np.mean(distances)) if distances else 0.0,
        'distances': distances,
        'n_centroids': len(centroids)
    }

def create_pca_visualization(embeddings, complexity_labels, complexity_scores, model_name):
    """Create PCA visualization of embeddings colored by complexity"""
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)
    
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                         c=complexity_scores, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter, label='Complexity Score')
    plt.xlabel(f'PC1 (explained variance: {pca.explained_variance_ratio_[0]:.3f})')
    plt.ylabel(f'PC2 (explained variance: {pca.explained_variance_ratio_[1]:.3f})')
    plt.title(f'PCA Visualization - {model_name}')
    plt.tight_layout()
    plt.savefig(f'/tmp/pca_{model_name.replace("/", "_")}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    return {
        'explained_variance_ratio': [float(x) for x in pca.explained_variance_ratio_],
        'total_explained_variance': float(sum(pca.explained_variance_ratio_))
    }

def main():
    start_time = time.time()
    print("Starting Syntactic Complexity Embedding Analysis")
    print("=" * 60)
    
    # Create controlled dataset
    print("Creating controlled sentence dataset...")
    dataset = create_controlled_sentence_dataset()
    print(f"Created dataset with {len(dataset)} sentences")
    
    # Check minimum sample size
    if len(dataset) < 50:
        print(f"ERROR: Dataset has only {len(dataset)} samples, minimum is 50")
        return
    
    # Extract data for analysis
    sentences = [item['sentence'] for item in dataset]
    semantic_contents = [item['semantic_content'] for item in dataset]
    complexity_levels = [item['complexity_level'] for item in dataset]
    complexity_scores = [item['complexity_score'] for item in dataset]
    
    print(f"Complexity score range: {min(complexity_scores):.2f} - {max(complexity_scores):.2f}")
    print(f"Complexity levels: {set(complexity_levels)}")
    print(f"Semantic contents: {len(set(semantic_contents))} unique")
    
    # Models to test
    models = ["all-MiniLM-L6-v2", "paraphrase-MiniLM-L3-v2"]
    
    results = {
        'experiment_info': {
            'dataset_size': len(dataset),
            'models_tested': models,
            'complexity_metrics': ['parse_depth', 'clause_count', 'dependency_distance'],
            'semantic_categories': len(set(semantic_contents))
        },
        'model_results': {},
        'summary': {}
    }
    
    print("\nTesting models...")
    print("-" * 40)
    
    for model_name in models:
        print(f"\nProcessing {model_name}...")
        
        # Get embeddings
        embeddings = get_embeddings(sentences, model_name)
        print(f"Embedding shape: {embeddings.shape}")
        
        # Variance analysis
        var_ratio, high_var, low_var = calculate_variance_metrics(embeddings, complexity_scores)
        print(f"Variance ratio (high/low complexity): {var_ratio:.4f}")
        
        # Clustering quality with permutation test
        print("Running permutation test...")
        perm_results = run_permutation_test(embeddings, complexity_levels, n_permutations=100)
        print(f"True silhouette score: {perm_results['true_score']:.4f}")
        print(f"Permutation test p-value: {perm_results['p_value']:.4f}")
        print(f"Significant: {perm_results['significant']}")
        
        # Supervised prediction
        print("Testing supervised complexity prediction...")
        sup_results = supervised_complexity_prediction(embeddings, complexity_scores)
        print(f"Classification accuracy: {sup_results['accuracy']:.4f}")
        
        # Centroid distance analysis
        centroid_results = centroid_distance_analysis(embeddings, complexity_levels)
        print(f"Mean centroid distance: {centroid_results['mean_distance']:.4f}")
        
        # PCA visualization
        pca_results = create_pca_visualization(embeddings, complexity_levels, 
                                             complexity_scores, model_name)
        print(f"PCA explained variance: {pca_results['total_explained_variance']:.4f}")
        
        # Store results
        results['model_results'][model_name] = {
            'variance_analysis': {
                'variance_ratio': var_ratio,
                'high_complexity_variance': high_var,
                'low_complexity_variance': low_var
            },
            'clustering_quality': perm_results,
            'supervised_prediction': sup_results,
            'centroid_analysis': centroid_results,
            'pca_analysis': pca_results,
            'embedding_stats': {
                'shape': list(embeddings.shape),
                'mean_norm': float(np.mean(np.linalg.norm(embeddings, axis=1))),
                'std_norm': float(np.std(np.linalg.norm(embeddings, axis=1)))
            }
        }
    
    # Summary analysis
    print("\n" + "="*60)
    print("SUMMARY RESULTS")
    print("="*60)
    
    summary_table = []
    summary_table.append(["Model", "Var Ratio", "Silhouette", "P-value", "Sig?", "Sup Acc", "Centroid Dist"])
    summary_table.append(["-"*20, "-"*9, "-"*10, "-"*8, "-"*4, "-"*7, "-"*12])
    
    significant_models = []
    
    for model_name in models:
        model_results = results['model_results'][model_name]
        var_ratio = model_results['variance_analysis']['variance_ratio']
        silhouette = model_results['clustering_quality']['true_score']
        p_value = model_results['clustering_quality']['p_value']
        significant = model_results['clustering_quality']['significant']
        sup_acc = model_results['supervised_prediction']['accuracy']
        centroid_dist = model_results['centroid_analysis']['mean_distance']
        
        if significant:
            significant_models.append(model_name)
        
        summary_table.append([
            model_name,
            f"{var_ratio:.3f}",
            f"{silhouette:.3f}",
            f"{p_value:.3f}",
            "Yes" if significant else "No",
            f"{sup_acc:.3f}",
            f"{centroid_dist:.3f}"
        ])
    
    # Print summary table
    for row in summary_table:
        print(f"{row[0]:<20} {row[1]:>9} {row[2]:>10} {row[3]:>8} {row[4]:>4} {row[5]:>7} {row[6]:>12}")
    
    # Overall conclusions
    print(f"\nModels with significant clustering: {significant_models}")
    print(f"Hypothesis supported: {'Yes' if significant_models else 'No'}")
    
    # Store summary
    results['summary'] = {
        'significant_models': significant_models,
        'hypothesis_supported': len(significant_models) > 0,
        'mean_variance_ratio': float(np.mean([results['model_results'][m]['variance_analysis']['variance_ratio'] for m in models])),
        'mean_silhouette_score': float(np.mean([results['model_results'][m]['clustering_quality']['true_score'] for m in models])),
        'mean_supervised_accuracy': float(np.mean([results['model_results'][m]['supervised_prediction']['accuracy'] for m in models]))
    }
    
    # Save results
    results_path = '/Users/kumacmini/cost-aware-research-search/results/iter_02_results.json'
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_path}")
    print(f"Total runtime: {time.time() - start_time:.2f} seconds")
    
    return results

if __name__ == "__main__":
    try:
        results = main()
        print("\nExperiment completed successfully!")
    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()