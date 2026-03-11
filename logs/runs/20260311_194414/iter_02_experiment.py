import json
import os
import sys
import time
import math
import random
import re
import collections
import statistics
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from scipy.stats import permutation_test
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
import openai

def get_api_client():
    """Initialize OpenAI client with API key from environment"""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    return openai.OpenAI(api_key=api_key)

def generate_controlled_sentences(client, n_samples=50):
    """Generate semantically controlled sentences with varying syntactic complexity"""
    print("Generating controlled sentence pairs...")
    
    base_meanings = [
        "A person reads a book",
        "The cat sleeps peacefully", 
        "Students learn new concepts",
        "The rain falls gently",
        "Children play in the park",
        "The teacher explains the lesson",
        "Birds fly across the sky",
        "The car moves down the street",
        "Workers build a house",
        "The dog chases the ball",
        "People walk through the forest",
        "The sun shines brightly",
        "Musicians play beautiful music",
        "The wind blows through trees"
    ]
    
    sentences = []
    api_calls = 0
    max_calls = 25
    
    for base_meaning in base_meanings:
        if api_calls >= max_calls:
            break
            
        try:
            # Generate sentences with different complexity levels
            prompt = f"""Generate exactly 3 sentences that express the same core meaning as "{base_meaning}" but with different syntactic complexity levels:

1. SIMPLE: Basic structure, minimal clauses
2. MEDIUM: One subordinate clause or compound structure  
3. COMPLEX: Multiple clauses, embedded structures, longer dependencies

Return only the 3 sentences, one per line, numbered 1-3."""

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=200,
                temperature=0.3
            )
            
            api_calls += 1
            lines = response.choices[0].message.content.strip().split('\n')
            
            for i, line in enumerate(lines):
                if line.strip() and any(char.isalpha() for char in line):
                    # Clean the sentence
                    sentence = re.sub(r'^\d+\.?\s*', '', line.strip())
                    if sentence:
                        sentences.append({
                            'sentence': sentence,
                            'meaning_group': base_meaning,
                            'complexity_level': i + 1  # 1=simple, 2=medium, 3=complex
                        })
            
        except Exception as e:
            print(f"Error generating sentences for '{base_meaning}': {e}")
            continue
    
    print(f"Generated {len(sentences)} sentences using {api_calls} API calls")
    return sentences

def calculate_syntax_metrics(sentence):
    """Calculate syntactic complexity metrics without external parsers"""
    words = sentence.split()
    word_count = len(words)
    
    # Parse depth approximation based on punctuation and conjunctions
    depth_indicators = sentence.count(',') + sentence.count(';') + sentence.count(':')
    subordinating_words = len([w for w in words if w.lower() in 
                              ['because', 'since', 'although', 'while', 'when', 'if', 'that', 'which', 'who']])
    parse_depth = 1 + depth_indicators + subordinating_words
    
    # Clause count approximation
    clause_markers = sentence.count(',') + sentence.count(';') + subordinating_words
    clause_count = max(1, 1 + clause_markers)
    
    # Dependency distance approximation (longer sentences = longer dependencies)
    avg_dependency_distance = word_count / 5.0  # Rough approximation
    
    return {
        'parse_depth': parse_depth,
        'clause_count': clause_count, 
        'dependency_distance': avg_dependency_distance,
        'word_count': word_count
    }

def get_embeddings(sentences, model_name):
    """Get sentence embeddings using specified model"""
    print(f"Getting embeddings with {model_name}...")
    model = SentenceTransformer(model_name)
    texts = [s['sentence'] for s in sentences]
    embeddings = model.encode(texts)
    return embeddings

def permutation_test_silhouette(embeddings, labels, n_permutations=100):
    """Perform permutation test for silhouette score significance"""
    true_score = silhouette_score(embeddings, labels)
    
    permuted_scores = []
    for _ in range(n_permutations):
        permuted_labels = np.random.permutation(labels)
        try:
            perm_score = silhouette_score(embeddings, permuted_labels)
            permuted_scores.append(perm_score)
        except:
            continue
    
    if not permuted_scores:
        return true_score, 1.0, np.array([])
    
    p_value = np.mean(np.array(permuted_scores) >= true_score)
    return true_score, p_value, np.array(permuted_scores)

def supervised_complexity_prediction(embeddings, complexity_labels):
    """Test supervised prediction of complexity from embeddings"""
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            embeddings, complexity_labels, test_size=0.3, random_state=42, stratify=complexity_labels
        )
        
        clf = LogisticRegression(random_state=42, max_iter=1000)
        clf.fit(X_train, y_train)
        
        train_acc = clf.score(X_train, y_train)
        test_acc = clf.score(X_test, y_test)
        
        return {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'n_train': len(X_train),
            'n_test': len(X_test)
        }
    except Exception as e:
        return {'error': str(e)}

def analyze_centroid_distances(embeddings, labels):
    """Analyze distances between complexity level centroids"""
    unique_labels = np.unique(labels)
    centroids = {}
    
    for label in unique_labels:
        mask = labels == label
        if np.sum(mask) > 0:
            centroids[label] = np.mean(embeddings[mask], axis=0)
    
    distances = {}
    for i, label1 in enumerate(unique_labels):
        for j, label2 in enumerate(unique_labels):
            if i < j and label1 in centroids and label2 in centroids:
                dist = np.linalg.norm(centroids[label1] - centroids[label2])
                distances[f"{label1}_{label2}"] = dist
    
    return distances

def run_experiment():
    """Run the complete experiment"""
    print("Starting syntactic complexity clustering experiment...")
    start_time = time.time()
    
    # Initialize
    client = get_api_client()
    results = {}
    
    try:
        # Generate controlled sentences
        sentences = generate_controlled_sentences(client, n_samples=75)
        
        if len(sentences) < 50:
            print(f"Warning: Only generated {len(sentences)} sentences, below minimum of 50")
        
        # Calculate syntax metrics
        print("Calculating syntactic complexity metrics...")
        for sentence in sentences:
            metrics = calculate_syntax_metrics(sentence['sentence'])
            sentence.update(metrics)
        
        # Models to test
        models = ["all-MiniLM-L6-v2", "paraphrase-MiniLM-L3-v2"]
        
        results['experiment_info'] = {
            'n_sentences': len(sentences),
            'n_models': len(models),
            'models_tested': models,
            'complexity_metrics': ['parse_depth', 'clause_count', 'dependency_distance']
        }
        
        results['sentences'] = sentences
        model_results = {}
        
        for model_name in models:
            print(f"\nTesting model: {model_name}")
            model_result = {}
            
            # Get embeddings
            embeddings = get_embeddings(sentences, model_name)
            
            # Test different complexity metrics
            for metric in ['parse_depth', 'clause_count', 'dependency_distance']:
                print(f"  Analyzing {metric}...")
                
                labels = np.array([s[metric] for s in sentences])
                
                # Convert to categorical if needed
                if metric in ['parse_depth', 'clause_count']:
                    # Discretize into bins for clustering
                    labels = np.digitize(labels, bins=np.percentile(labels, [33, 67]))
                else:
                    # For continuous metrics, use tertiles
                    labels = np.digitize(labels, bins=np.percentile(labels, [33, 67]))
                
                # Skip if not enough unique labels
                if len(np.unique(labels)) < 2:
                    continue
                
                metric_result = {}
                
                # Clustering analysis
                try:
                    # Silhouette analysis with permutation test
                    true_sil, p_value, perm_scores = permutation_test_silhouette(
                        embeddings, labels, n_permutations=50
                    )
                    
                    metric_result['silhouette'] = {
                        'score': float(true_sil),
                        'p_value': float(p_value),
                        'permutation_mean': float(np.mean(perm_scores)) if len(perm_scores) > 0 else 0.0,
                        'permutation_std': float(np.std(perm_scores)) if len(perm_scores) > 0 else 0.0
                    }
                    
                    # Variance analysis
                    within_variance = []
                    between_variance = []
                    
                    for label_val in np.unique(labels):
                        mask = labels == label_val
                        if np.sum(mask) > 1:
                            group_embeddings = embeddings[mask]
                            within_var = np.mean(np.var(group_embeddings, axis=0))
                            within_variance.append(within_var)
                    
                    total_var = np.mean(np.var(embeddings, axis=0))
                    within_var_mean = np.mean(within_variance) if within_variance else total_var
                    
                    metric_result['variance'] = {
                        'within_group_variance': float(within_var_mean),
                        'total_variance': float(total_var),
                        'variance_ratio': float(within_var_mean / total_var) if total_var > 0 else 1.0
                    }
                    
                    # Supervised prediction
                    sup_result = supervised_complexity_prediction(embeddings, labels)
                    metric_result['supervised_prediction'] = sup_result
                    
                    # Centroid distances
                    centroid_distances = analyze_centroid_distances(embeddings, labels)
                    metric_result['centroid_distances'] = centroid_distances
                    
                except Exception as e:
                    metric_result['error'] = str(e)
                
                model_result[metric] = metric_result
            
            model_results[model_name] = model_result
        
        results['model_results'] = model_results
        
        # Summary statistics
        print("\n=== EXPERIMENT SUMMARY ===")
        print(f"Total sentences analyzed: {len(sentences)}")
        print(f"Meaning groups: {len(set(s['meaning_group'] for s in sentences))}")
        print(f"Runtime: {time.time() - start_time:.2f} seconds")
        
        summary_table = []
        
        for model_name in models:
            for metric in ['parse_depth', 'clause_count', 'dependency_distance']:
                if metric in model_results[model_name]:
                    result = model_results[model_name][metric]
                    if 'silhouette' in result:
                        sil = result['silhouette']
                        var = result['variance']
                        sup = result['supervised_prediction']
                        
                        summary_table.append({
                            'model': model_name,
                            'metric': metric,
                            'silhouette_score': sil['score'],
                            'silhouette_p_value': sil['p_value'],
                            'variance_ratio': var['variance_ratio'],
                            'supervised_accuracy': sup.get('test_accuracy', 0.0)
                        })
        
        results['summary_table'] = summary_table
        
        print("\nModel\t\tMetric\t\t\tSilhouette\tP-value\t\tVar.Ratio\tSup.Acc")
        print("-" * 90)
        for row in summary_table:
            print(f"{row['model'][:15]:<15}\t{row['metric'][:15]:<15}\t{row['silhouette_score']:.3f}\t\t{row['p_value']:.3f}\t\t{row['variance_ratio']:.3f}\t\t{row['supervised_accuracy']:.3f}")
        
        # Significance test results
        significant_results = [r for r in summary_table if r['silhouette_p_value'] < 0.05 and r['silhouette_score'] > 0.05]
        
        print(f"\nSignificant results (p < 0.05, silhouette > 0.05): {len(significant_results)}")
        
        results['significant_results'] = significant_results
        results['success'] = len(significant_results) > 0
        results['runtime_seconds'] = time.time() - start_time
        
        # Save results
        output_path = Path("/Users/kumacmini/cost-aware-research-search/results/iter_02_results.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")
        print(f"Experiment completed successfully: {results['success']}")
        
    except Exception as e:
        print(f"Experiment failed: {e}")
        results['error'] = str(e)
        results['success'] = False
        
        # Save error results
        output_path = Path("/Users/kumacmini/cost-aware-research-search/results/iter_02_results.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

if __name__ == "__main__":
    run_experiment()