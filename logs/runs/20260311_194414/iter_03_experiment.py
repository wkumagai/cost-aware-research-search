import json
import os
import sys
import time
import math
import random
import re
import statistics
from collections import defaultdict, Counter
from datetime import datetime
import numpy as np
from scipy import stats
from sklearn.model_selection import GroupKFold, cross_val_score, permutation_test_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
import openai

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

# Initialize OpenAI client
openai.api_key = os.getenv('OPENAI_API_KEY')

def create_syntactic_complexity_data():
    """Create paraphrase pairs with controlled syntactic complexity differences"""
    
    # Base semantic templates with paraphrase pairs
    templates = [
        # Simple actions
        {
            "semantic_category": "action_simple",
            "simple": "The cat sleeps on the mat.",
            "complex": "The cat, which is tired from playing, sleeps peacefully on the soft mat that sits by the window."
        },
        {
            "semantic_category": "action_simple", 
            "simple": "Dogs bark loudly.",
            "complex": "The neighborhood dogs, responding to distant sirens, bark loudly in unison while their owners try to calm them."
        },
        {
            "semantic_category": "action_simple",
            "simple": "Children play games.",
            "complex": "Young children, excited by the sunny weather, play imaginative games that their parents taught them last summer."
        },
        {
            "semantic_category": "action_simple",
            "simple": "Birds fly south.",
            "complex": "Migratory birds, following ancient instincts, fly south in perfect formations when winter approaches the northern regions."
        },
        {
            "semantic_category": "action_simple",
            "simple": "Rain falls heavily.",
            "complex": "Heavy rain, driven by strong winds from the ocean, falls steadily on the city streets while people rush for shelter."
        },
        
        # Relationships
        {
            "semantic_category": "relationship",
            "simple": "Sarah loves reading books.",
            "complex": "Sarah, who inherited her love of literature from her grandmother, loves reading classic books that challenge her thinking."
        },
        {
            "semantic_category": "relationship",
            "simple": "Teachers help students learn.",
            "complex": "Dedicated teachers, drawing on years of experience, help struggling students learn difficult concepts through patient guidance."
        },
        {
            "semantic_category": "relationship",
            "simple": "Friends meet for coffee.",
            "complex": "Old friends, who haven't seen each other in months, meet for coffee at the cafe where they first became acquainted."
        },
        {
            "semantic_category": "relationship",
            "simple": "Parents support their children.",
            "complex": "Loving parents, despite their own challenges, support their children through difficult decisions that will shape their futures."
        },
        {
            "semantic_category": "relationship",
            "simple": "Neighbors help each other.",
            "complex": "Friendly neighbors, recognizing the importance of community, help each other during times when assistance is most needed."
        },
        
        # States/descriptions
        {
            "semantic_category": "state_description",
            "simple": "The house looks beautiful.",
            "complex": "The renovated house, painted in warm colors chosen carefully, looks beautiful against the backdrop of mature trees."
        },
        {
            "semantic_category": "state_description",
            "simple": "Food tastes delicious.",
            "complex": "The carefully prepared food, seasoned with herbs from the garden, tastes delicious to everyone who tries it."
        },
        {
            "semantic_category": "state_description",
            "simple": "Music sounds pleasant.",
            "complex": "The classical music, performed by talented musicians, sounds pleasant to listeners who appreciate artistic expression."
        },
        {
            "semantic_category": "state_description",
            "simple": "Weather feels warm.",
            "complex": "The spring weather, warmed by gentle sunshine, feels wonderfully warm to people who endured the harsh winter."
        },
        {
            "semantic_category": "state_description",
            "simple": "Flowers smell fragrant.",
            "complex": "The garden flowers, blooming in vibrant colors, smell intensely fragrant in the morning air that carries their scent."
        },
        
        # Processes
        {
            "semantic_category": "process",
            "simple": "Water boils quickly.",
            "complex": "The pot of water, heated on the efficient stove, boils quickly while steam rises and fills the kitchen."
        },
        {
            "semantic_category": "process",
            "simple": "Plants grow tall.",
            "complex": "The healthy plants, nourished by rich soil and regular watering, grow surprisingly tall throughout the growing season."
        },
        {
            "semantic_category": "process",
            "simple": "Ice melts slowly.",
            "complex": "The thick ice, formed during the coldest nights, melts slowly in the warming sun that signals spring's arrival."
        },
        {
            "semantic_category": "process",
            "simple": "Bread bakes evenly.",
            "complex": "The homemade bread, mixed with traditional ingredients, bakes evenly in the oven while filling the house with aroma."
        },
        {
            "semantic_category": "process",
            "simple": "Cars move forward.",
            "complex": "The line of cars, waiting for the traffic signal, moves forward cautiously when the light turns green."
        },
        
        # Events
        {
            "semantic_category": "event",
            "simple": "Concerts happen weekly.",
            "complex": "The popular concerts, featuring local and visiting artists, happen weekly at the venue that hosts cultural events."
        },
        {
            "semantic_category": "event",
            "simple": "Markets open early.",
            "complex": "The bustling farmers markets, offering fresh local produce, open early when vendors arrive with goods from nearby farms."
        },
        {
            "semantic_category": "event",
            "simple": "Schools close temporarily.",
            "complex": "The district schools, following safety protocols, close temporarily when weather conditions become dangerous for transportation."
        },
        {
            "semantic_category": "event",
            "simple": "Festivals celebrate culture.",
            "complex": "The annual festivals, organized by dedicated volunteers, celebrate local culture through music, food, and traditional activities."
        },
        {
            "semantic_category": "event",
            "simple": "Games start promptly.",
            "complex": "The scheduled games, anticipated by enthusiastic fans, start promptly when teams finish their preparation and warm-up routines."
        }
    ]
    
    return templates

def calculate_complexity_metrics(sentence):
    """Calculate syntactic complexity metrics using rule-based approximations"""
    # Parse depth approximation (nested structures)
    parse_depth = 1
    depth_indicators = ['which', 'that', 'who', 'where', 'when', 'while', 'although', 'because', 'since', 'if']
    for indicator in depth_indicators:
        if indicator in sentence.lower():
            parse_depth += 1
    
    # Add depth for comma-separated clauses
    commas = sentence.count(',')
    parse_depth += min(commas // 2, 3)  # Cap at reasonable level
    
    # Clause count approximation
    clause_markers = [',', 'and', 'but', 'or', 'which', 'that', 'who', 'when', 'where', 'while', 'because', 'since', 'although', 'if']
    clause_count = 1  # Base clause
    for marker in clause_markers:
        clause_count += sentence.lower().count(marker)
    
    # Dependency distance approximation (distance between related words)
    words = sentence.split()
    dependency_distance = 0
    
    # Look for long-distance dependencies
    for i, word in enumerate(words):
        word_lower = word.lower().strip('.,!?')
        # Relative pronouns create long dependencies
        if word_lower in ['which', 'that', 'who', 'where', 'when']:
            # Distance to likely antecedent
            dependency_distance += min(i, 5)  # Cap at 5
        
        # Prepositions can create dependencies
        if word_lower in ['of', 'in', 'on', 'at', 'by', 'for', 'with', 'from']:
            dependency_distance += 1
    
    # Normalize by sentence length
    if len(words) > 0:
        dependency_distance = dependency_distance / len(words) * 10  # Scale for readability
    
    return {
        'parse_depth': parse_depth,
        'clause_count': clause_count, 
        'dependency_distance': dependency_distance
    }

def compute_embeddings(sentences, model_name):
    """Compute embeddings for sentences using specified model"""
    print(f"Computing embeddings with {model_name}...")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(sentences)
    return embeddings

def verify_semantic_similarity(embedding1, embedding2, threshold=0.7):
    """Verify that paraphrase pairs have high semantic similarity"""
    similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
    return similarity >= threshold

def create_dataset():
    """Create complete dataset with embeddings and complexity metrics"""
    templates = create_syntactic_complexity_data()
    
    # Create sentence pairs
    sentences = []
    complexity_labels = []
    groups = []
    complexity_metrics = []
    
    for template in templates:
        simple_sent = template['simple']
        complex_sent = template['complex']
        
        # Calculate complexity metrics
        simple_metrics = calculate_complexity_metrics(simple_sent)
        complex_metrics = calculate_complexity_metrics(complex_sent)
        
        sentences.extend([simple_sent, complex_sent])
        complexity_labels.extend([0, 1])  # 0 = simple, 1 = complex
        groups.extend([template['semantic_category'], template['semantic_category']])
        complexity_metrics.extend([simple_metrics, complex_metrics])
    
    return sentences, complexity_labels, groups, complexity_metrics

def analyze_complexity_correlations(complexity_metrics):
    """Analyze correlations between different complexity metrics"""
    parse_depths = [m['parse_depth'] for m in complexity_metrics]
    clause_counts = [m['clause_count'] for m in complexity_metrics]
    dependency_distances = [m['dependency_distance'] for m in complexity_metrics]
    
    correlations = {
        'parse_depth_clause_count': stats.pearsonr(parse_depths, clause_counts)[0],
        'parse_depth_dependency_distance': stats.pearsonr(parse_depths, dependency_distances)[0],
        'clause_count_dependency_distance': stats.pearsonr(clause_counts, dependency_distances)[0]
    }
    
    return correlations

def run_grouped_cross_validation(embeddings, labels, groups, n_splits=5):
    """Run grouped cross-validation to prevent data leakage"""
    clf = LogisticRegression(random_state=42, max_iter=1000)
    scaler = StandardScaler()
    
    gkf = GroupKFold(n_splits=n_splits)
    scores = []
    
    for train_idx, test_idx in gkf.split(embeddings, labels, groups):
        X_train, X_test = embeddings[train_idx], embeddings[test_idx]
        y_train, y_test = np.array(labels)[train_idx], np.array(labels)[test_idx]
        
        # Scale features
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train and evaluate
        clf.fit(X_train_scaled, y_train)
        score = clf.score(X_test_scaled, y_test)
        scores.append(score)
    
    return scores

def run_permutation_baseline(embeddings, labels, groups, n_permutations=100):
    """Run permutation test to establish baseline performance"""
    clf = LogisticRegression(random_state=42, max_iter=1000)
    scaler = StandardScaler()
    
    # Scale embeddings
    embeddings_scaled = scaler.fit_transform(embeddings)
    
    # Run permutation test with grouped CV
    gkf = GroupKFold(n_splits=3)  # Fewer splits for permutation test
    
    score, perm_scores, pvalue = permutation_test_score(
        clf, embeddings_scaled, labels, groups=groups, cv=gkf,
        n_permutations=n_permutations, random_state=42
    )
    
    return score, perm_scores, pvalue

def analyze_distance_patterns(embeddings, labels, groups):
    """Analyze within-category vs between-complexity distance patterns"""
    results = {}
    
    # Group embeddings by category and complexity
    category_embeddings = defaultdict(lambda: {'simple': [], 'complex': []})
    
    for i, (emb, label, group) in enumerate(zip(embeddings, labels, groups)):
        complexity_type = 'simple' if label == 0 else 'complex'
        category_embeddings[group][complexity_type].append(emb)
    
    within_category_distances = []
    between_complexity_distances = []
    
    for category, embs in category_embeddings.items():
        simple_embs = np.array(embs['simple'])
        complex_embs = np.array(embs['complex'])
        
        if len(simple_embs) > 0 and len(complex_embs) > 0:
            # Within-category, between-complexity distances
            for s_emb in simple_embs:
                for c_emb in complex_embs:
                    dist = np.linalg.norm(s_emb - c_emb)
                    within_category_distances.append(dist)
    
    # Between-category distances (same complexity)
    categories = list(category_embeddings.keys())
    for i in range(len(categories)):
        for j in range(i+1, len(categories)):
            cat1, cat2 = categories[i], categories[j]
            
            # Simple-simple across categories
            for s1 in category_embeddings[cat1]['simple']:
                for s2 in category_embeddings[cat2]['simple']:
                    dist = np.linalg.norm(s1 - s2)
                    between_complexity_distances.append(dist)
            
            # Complex-complex across categories  
            for c1 in category_embeddings[cat1]['complex']:
                for c2 in category_embeddings[cat2]['complex']:
                    dist = np.linalg.norm(c1 - c2)
                    between_complexity_distances.append(dist)
    
    results['within_category_mean'] = np.mean(within_category_distances) if within_category_distances else 0
    results['within_category_std'] = np.std(within_category_distances) if within_category_distances else 0
    results['between_complexity_mean'] = np.mean(between_complexity_distances) if between_complexity_distances else 0
    results['between_complexity_std'] = np.std(between_complexity_distances) if between_complexity_distances else 0
    
    # Variance ratio
    if results['between_complexity_std'] > 0:
        results['variance_ratio'] = results['within_category_std'] / results['between_complexity_std']
    else:
        results['variance_ratio'] = float('inf')
    
    return results

def calculate_effect_size(group1, group2):
    """Calculate Cohen's d effect size"""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    
    if pooled_std == 0:
        return 0
    
    cohen_d = (np.mean(group1) - np.mean(group2)) / pooled_std
    return cohen_d

def main():
    start_time = time.time()
    print("Starting syntactic complexity detection experiment...")
    
    # Create dataset
    print("Creating dataset...")
    sentences, labels, groups, complexity_metrics = create_dataset()
    print(f"Created dataset with {len(sentences)} sentences across {len(set(groups))} semantic categories")
    
    # Analyze complexity metric correlations
    correlations = analyze_complexity_correlations(complexity_metrics)
    print(f"Complexity metric correlations: {correlations}")
    
    # Models to test
    models = ["all-MiniLM-L6-v2", "paraphrase-MiniLM-L3-v2", "all-mpnet-base-v2"]
    
    results = {
        'experiment_metadata': {
            'timestamp': datetime.now().isoformat(),
            'n_samples': len(sentences),
            'n_categories': len(set(groups)),
            'models_tested': models,
            'complexity_correlations': correlations
        },
        'model_results': {}
    }
    
    # Test each model
    for model_name in models:
        print(f"\nTesting model: {model_name}")
        model_results = {}
        
        try:
            # Compute embeddings
            embeddings = compute_embeddings(sentences, model_name)
            print(f"Computed embeddings shape: {embeddings.shape}")
            
            # Verify semantic similarity for paraphrase pairs
            similarity_scores = []
            valid_pairs = 0
            for i in range(0, len(embeddings), 2):
                if i+1 < len(embeddings):
                    sim = verify_semantic_similarity(embeddings[i], embeddings[i+1])
                    similarity_scores.append(np.dot(embeddings[i], embeddings[i+1]) / 
                                           (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i+1])))
                    if sim:
                        valid_pairs += 1
            
            model_results['semantic_similarity'] = {
                'mean': float(np.mean(similarity_scores)),
                'std': float(np.std(similarity_scores)),
                'valid_pairs_ratio': valid_pairs / (len(embeddings) // 2)
            }
            
            # Grouped cross-validation
            cv_scores = run_grouped_cross_validation(embeddings, labels, groups)
            model_results['grouped_cv'] = {
                'scores': [float(s) for s in cv_scores],
                'mean': float(np.mean(cv_scores)),
                'std': float(np.std(cv_scores)),
                'confidence_interval': [
                    float(np.mean(cv_scores) - 1.96 * np.std(cv_scores) / np.sqrt(len(cv_scores))),
                    float(np.mean(cv_scores) + 1.96 * np.std(cv_scores) / np.sqrt(len(cv_scores)))
                ]
            }
            
            # Permutation baseline
            true_score, perm_scores, p_value = run_permutation_baseline(embeddings, labels, groups, n_permutations=50)
            model_results['permutation_baseline'] = {
                'true_score': float(true_score),
                'permutation_scores_mean': float(np.mean(perm_scores)),
                'permutation_scores_std': float(np.std(perm_scores)),
                'p_value': float(p_value),
                'score_above_baseline': float(true_score - np.mean(perm_scores))
            }
            
            # Distance analysis
            distance_results = analyze_distance_patterns(embeddings, labels, groups)
            model_results['distance_analysis'] = {k: float(v) for k, v in distance_results.items()}
            
            # Effect sizes for complexity metrics
            simple_indices = [i for i, label in enumerate(labels) if label == 0]
            complex_indices = [i for i, label in enumerate(labels) if label == 1]
            
            model_results['complexity_effect_sizes'] = {}
            
            for metric in ['parse_depth', 'clause_count', 'dependency_distance']:
                simple_values = [complexity_metrics[i][metric] for i in simple_indices]
                complex_values = [complexity_metrics[i][metric] for i in complex_indices]
                
                effect_size = calculate_effect_size(complex_values, simple_values)
                model_results['complexity_effect_sizes'][metric] = {
                    'cohens_d': float(effect_size),
                    'simple_mean': float(np.mean(simple_values)),
                    'complex_mean': float(np.mean(complex_values)),
                    'simple_std': float(np.std(simple_values)),
                    'complex_std': float(np.std(complex_values))
                }
            
            results['model_results'][model_name] = model_results
            
        except Exception as e:
            print(f"Error with model {model_name}: {e}")
            results['model_results'][model_name] = {'error': str(e)}
    
    # Summary analysis
    successful_models = [m for m in models if 'error' not in results['model_results'][m]]
    if successful_models:
        best_model = max(successful_models, 
                        key=lambda m: results['model_results'][m]['grouped_cv']['mean'])
        results['summary'] = {
            'best_model': best_model,
            'best_accuracy': results['model_results'][best_model]['grouped_cv']['mean'],
            'significant_above_baseline': [],
            'effect_sizes_summary': {}
        }
        
        # Check significance above baseline
        for model in successful_models:
            model_res = results['model_results'][model]
            if model_res['permutation_baseline']['p_value'] < 0.05:
                results['summary']['significant_above_baseline'].append(model)
        
        # Aggregate effect sizes
        for metric in ['parse_depth', 'clause_count', 'dependency_distance']:
            effect_sizes = []
            for model in successful_models:
                if 'complexity_effect_sizes' in results['model_results'][model]:
                    effect_sizes.append(results['model_results'][model]['complexity_effect_sizes'][metric]['cohens_d'])
            
            if effect_sizes:
                results['summary']['effect_sizes_summary'][metric] = {
                    'mean': float(np.mean(effect_sizes)),
                    'std': float(np.std(effect_sizes)),
                    'models_with_large_effect': len([es for es in effect_sizes if abs(es) > 0.5])
                }
    
    # Save results
    output_path = "/Users/kumacmini/cost-aware-research-search/results/iter_03_results.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary table
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)
    
    if successful_models:
        print(f"Models tested successfully: {len(successful_models)}")
        print(f"Best model: {results['summary']['best_model']}")
        print(f"Best accuracy: {results['summary']['best_accuracy']:.3f}")
        print(f"Models significantly above baseline: {len(results['summary']['significant_above_baseline'])}")
        
        print("\nModel Performance:")
        for model in successful_models:
            model_res = results['model_results'][model]
            acc = model_res['grouped_cv']['mean']
            p_val = model_res['permutation_baseline']['p_value']
            above_baseline = model_res['permutation_baseline']['score_above_baseline']
            print(f"  {model}: {acc:.3f} ± {model_res['grouped_cv']['std']:.3f}, "
                  f"above baseline: +{above_baseline:.3f}, p={p_val:.3f}")
        
        print("\nEffect Sizes by Complexity Metric:")
        for metric, summary in results['summary']['effect_sizes_summary'].items():
            print(f"  {metric}: Cohen's d = {summary['mean']:.3f} ± {summary['std']:.3f}, "
                  f"large effects: {summary['models_with_large_effect']}/{len(successful_models)}")
    
    print(f"\nComplexity Metric Correlations:")
    for pair, corr in correlations.items():
        print(f"  {pair}: r = {corr:.3f}")
    
    runtime = time.time() - start_time
    print(f"\nExperiment completed in {runtime:.1f} seconds")
    print(f"Results saved to: {output_path}")

if __name__ == "__main__":
    main()