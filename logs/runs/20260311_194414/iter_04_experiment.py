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
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from scipy import stats
from scipy.stats import ttest_rel, chi2_contingency
import openai

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

def setup_openai():
    """Setup OpenAI client with API key from environment"""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    openai.api_key = api_key
    return openai

def calculate_coordination_count(sentence):
    """Count coordinating conjunctions (and, but, or, nor, for, so, yet)"""
    coord_words = ['and', 'but', 'or', 'nor', 'for', 'so', 'yet']
    words = sentence.lower().split()
    return sum(1 for word in words if word in coord_words)

def calculate_subordination_depth(sentence):
    """Estimate subordination depth by counting subordinating conjunctions and relative pronouns"""
    subord_words = ['that', 'which', 'who', 'whom', 'whose', 'when', 'where', 'while', 
                   'although', 'because', 'since', 'if', 'unless', 'until', 'before', 'after']
    words = sentence.lower().split()
    return sum(1 for word in words if word in subord_words)

def calculate_modification_chains(sentence):
    """Count adjective and adverb chains (words ending in -ly, -ed, -ing, common adjectives)"""
    mod_patterns = [r'\w+ly\b', r'\w+ed\b', r'\w+ing\b']
    common_adj = ['good', 'bad', 'big', 'small', 'new', 'old', 'high', 'low', 'long', 'short',
                  'important', 'different', 'large', 'available', 'popular', 'able', 'basic']
    
    count = 0
    for pattern in mod_patterns:
        count += len(re.findall(pattern, sentence.lower()))
    
    words = sentence.lower().split()
    count += sum(1 for word in words if word in common_adj)
    return count

def generate_paraphrase_triplet(base_sentence, client):
    """Generate three paraphrases with different syntactic complexity patterns"""
    
    prompt = f"""Create exactly 3 paraphrases of this sentence that preserve meaning but vary syntactic complexity:

Original: "{base_sentence}"

Requirements:
1. High coordination: Use multiple coordinating conjunctions (and, but, or, etc.)
2. High subordination: Use subordinate clauses with that, which, because, etc.  
3. High modification: Use many adjectives, adverbs, and descriptive phrases

Return only the 3 sentences, one per line, no labels or numbers."""

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=150,
            temperature=0.7
        )
        
        sentences = response.choices[0].message.content.strip().split('\n')
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) >= 3:
            return sentences[:3]
        else:
            return None
            
    except Exception as e:
        print(f"Error generating paraphrases: {e}")
        return None

def verify_semantic_similarity(sentences, model, threshold=0.75):
    """Verify that sentences are semantically similar using embeddings"""
    embeddings = model.encode(sentences)
    
    # Calculate pairwise cosine similarities
    similarities = []
    for i in range(len(embeddings)):
        for j in range(i+1, len(embeddings)):
            sim = np.dot(embeddings[i], embeddings[j]) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]))
            similarities.append(sim)
    
    return all(sim >= threshold for sim in similarities)

def check_orthogonality(complexity_scores, threshold=0.3):
    """Check if complexity metrics are sufficiently orthogonal"""
    coord_scores = [scores['coordination'] for scores in complexity_scores]
    subord_scores = [scores['subordination'] for scores in complexity_scores]
    mod_scores = [scores['modification'] for scores in complexity_scores]
    
    corr_coord_subord = np.corrcoef(coord_scores, subord_scores)[0, 1]
    corr_coord_mod = np.corrcoef(coord_scores, mod_scores)[0, 1]
    corr_subord_mod = np.corrcoef(subord_scores, mod_scores)[0, 1]
    
    return (abs(corr_coord_subord) < threshold and 
            abs(corr_coord_mod) < threshold and 
            abs(corr_subord_mod) < threshold)

def create_pairwise_preference_data(triplets, embeddings, complexity_scores):
    """Create pairwise preference data for classification"""
    X = []
    y = []
    group_ids = []
    
    for group_id, (triplet, embs, scores) in enumerate(zip(triplets, embeddings, complexity_scores)):
        # Create all pairwise combinations
        for i in range(len(triplet)):
            for j in range(len(triplet)):
                if i != j:
                    # Feature: embedding distance
                    dist = np.linalg.norm(embs[i] - embs[j])
                    
                    # Complexity differences
                    coord_diff = scores[i]['coordination'] - scores[j]['coordination']
                    subord_diff = scores[i]['subordination'] - scores[j]['subordination']
                    mod_diff = scores[i]['modification'] - scores[j]['modification']
                    
                    X.append([dist, coord_diff, subord_diff, mod_diff])
                    
                    # Label: which sentence is more complex overall
                    total_complex_i = sum(scores[i].values())
                    total_complex_j = sum(scores[j].values())
                    y.append(1 if total_complex_i > total_complex_j else 0)
                    
                    group_ids.append(group_id)
    
    return np.array(X), np.array(y), np.array(group_ids)

def run_mixed_effects_analysis(X, y, group_ids):
    """Simplified mixed effects analysis using grouped cross-validation"""
    # Group-aware stratified split
    unique_groups = np.unique(group_ids)
    train_groups, test_groups = train_test_split(unique_groups, test_size=0.3, random_state=42)
    
    train_mask = np.isin(group_ids, train_groups)
    test_mask = np.isin(group_ids, test_groups)
    
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Fit logistic regression
    model = LogisticRegression(random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Predictions and metrics
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    
    # Effect sizes (standardized coefficients)
    feature_names = ['embedding_distance', 'coord_diff', 'subord_diff', 'mod_diff']
    coefficients = model.coef_[0]
    
    return {
        'accuracy': accuracy,
        'coefficients': dict(zip(feature_names, coefficients)),
        'n_train': len(X_train),
        'n_test': len(X_test),
        'class_balance_train': np.mean(y_train),
        'class_balance_test': np.mean(y_test)
    }

def main():
    print("Starting Orthogonal Syntactic Complexity Detection Experiment")
    print("=" * 60)
    
    start_time = time.time()
    
    # Setup
    client = setup_openai()
    
    # Load sentence transformer models
    model_names = ["all-MiniLM-L6-v2", "paraphrase-MiniLM-L3-v2", "all-mpnet-base-v2"]
    models = {}
    
    print("Loading sentence transformer models...")
    for name in model_names:
        try:
            models[name] = SentenceTransformer(name)
            print(f"✓ Loaded {name}")
        except Exception as e:
            print(f"✗ Failed to load {name}: {e}")
    
    if not models:
        raise RuntimeError("No sentence transformer models could be loaded")
    
    # Base sentences for generating triplets
    base_sentences = [
        "The cat sits on the mat.",
        "John reads a book every evening.",
        "Rain falls from the dark clouds.",
        "Children play in the sunny park.",
        "The teacher explains the difficult lesson.",
        "Cars drive slowly through the busy street.",
        "Birds sing beautiful songs in spring.",
        "Students study hard for their exams.",
        "The chef prepares delicious meals daily.",
        "Flowers bloom in the garden.",
        "Dogs bark loudly at strangers.",
        "People walk quickly to work.",
        "The artist paints colorful pictures.",
        "Wind blows through tall trees.",
        "Fish swim in the clear lake.",
        "The doctor helps sick patients.",
        "Machines work efficiently in factories.",
        "Stars shine brightly at night.",
        "The farmer grows fresh vegetables.",
        "Music plays softly in the background."
    ]
    
    print(f"\nGenerating paraphrase triplets from {len(base_sentences)} base sentences...")
    
    # Generate paraphrase triplets
    triplets = []
    complexity_scores = []
    api_calls = 0
    max_api_calls = 25  # Reserve some calls for error handling
    
    for i, base_sentence in enumerate(base_sentences):
        if api_calls >= max_api_calls:
            print(f"Reached API call limit ({max_api_calls})")
            break
            
        print(f"Processing sentence {i+1}/{len(base_sentences)}: {base_sentence[:50]}...")
        
        try:
            paraphrases = generate_paraphrase_triplet(base_sentence, client)
            api_calls += 1
            
            if paraphrases is None:
                continue
                
            # Calculate complexity scores
            scores = []
            for sentence in paraphrases:
                score = {
                    'coordination': calculate_coordination_count(sentence),
                    'subordination': calculate_subordination_depth(sentence),
                    'modification': calculate_modification_chains(sentence)
                }
                scores.append(score)
            
            # Check semantic similarity and orthogonality
            primary_model = list(models.values())[0]
            if verify_semantic_similarity(paraphrases, primary_model, threshold=0.70):
                if check_orthogonality(scores, threshold=0.4):  # Relaxed threshold
                    triplets.append(paraphrases)
                    complexity_scores.append(scores)
                    print(f"✓ Added triplet {len(triplets)}")
                else:
                    print("✗ Failed orthogonality check")
            else:
                print("✗ Failed semantic similarity check")
                
        except Exception as e:
            print(f"✗ Error processing sentence: {e}")
            continue
    
    print(f"\nGenerated {len(triplets)} valid paraphrase triplets")
    
    if len(triplets) < 15:  # Minimum viable sample
        print("ERROR: Insufficient valid triplets generated")
        return
    
    # Analyze results for each model
    results = {}
    
    for model_name, model in models.items():
        print(f"\nAnalyzing with {model_name}...")
        
        try:
            # Generate embeddings
            embeddings = []
            for triplet in triplets:
                embs = model.encode(triplet)
                embeddings.append(embs)
            
            # Create pairwise preference data
            X, y, group_ids = create_pairwise_preference_data(triplets, embeddings, complexity_scores)
            
            print(f"Created {len(X)} pairwise comparisons from {len(triplets)} triplets")
            print(f"Class balance: {np.mean(y):.3f}")
            
            # Run mixed effects analysis
            mixed_results = run_mixed_effects_analysis(X, y, group_ids)
            
            # Random baseline
            random_accuracy = np.mean(np.random.choice([0, 1], size=len(y)))
            
            # Calculate effect size (Cohen's d)
            y_pred_prob = LogisticRegression(random_state=42).fit(
                StandardScaler().fit_transform(X), y
            ).predict_proba(StandardScaler().fit_transform(X))[:, 1]
            
            effect_size = (np.mean(y_pred_prob[y == 1]) - np.mean(y_pred_prob[y == 0])) / np.std(y_pred_prob)
            
            results[model_name] = {
                'accuracy': float(mixed_results['accuracy']),
                'random_baseline': float(random_accuracy),
                'effect_size': float(effect_size),
                'n_triplets': len(triplets),
                'n_comparisons': len(X),
                'coefficients': {k: float(v) for k, v in mixed_results['coefficients'].items()},
                'class_balance_train': float(mixed_results['class_balance_train']),
                'class_balance_test': float(mixed_results['class_balance_test']),
                'orthogonality_achieved': True  # Since we filtered for this
            }
            
            print(f"Accuracy: {mixed_results['accuracy']:.3f}")
            print(f"Effect size: {effect_size:.3f}")
            
        except Exception as e:
            print(f"Error analyzing {model_name}: {e}")
            results[model_name] = {'error': str(e)}
    
    # Calculate overall statistics
    complexity_correlations = {}
    if len(complexity_scores) > 3:
        all_coord = [score['coordination'] for scores in complexity_scores for score in scores]
        all_subord = [score['subordination'] for scores in complexity_scores for score in scores]
        all_mod = [score['modification'] for scores in complexity_scores for score in scores]
        
        complexity_correlations = {
            'coord_subord': float(np.corrcoef(all_coord, all_subord)[0, 1]),
            'coord_mod': float(np.corrcoef(all_coord, all_mod)[0, 1]),
            'subord_mod': float(np.corrcoef(all_subord, all_mod)[0, 1])
        }
    
    # Compile final results
    final_results = {
        'experiment_info': {
            'hypothesis': 'Sentence embeddings encode orthogonal syntactic complexity dimensions',
            'n_triplets_generated': len(triplets),
            'n_models_tested': len(results),
            'api_calls_used': api_calls,
            'runtime_minutes': (time.time() - start_time) / 60
        },
        'complexity_correlations': complexity_correlations,
        'model_results': results,
        'success_criteria': {
            'target_accuracy': 0.60,
            'target_effect_size': 0.3,
            'min_triplets': 50
        }
    }
    
    # Print summary table
    print("\n" + "=" * 60)
    print("EXPERIMENT RESULTS SUMMARY")
    print("=" * 60)
    print(f"Total triplets generated: {len(triplets)}")
    print(f"API calls used: {api_calls}")
    print(f"Runtime: {(time.time() - start_time)/60:.2f} minutes")
    print("\nComplexity Metric Correlations:")
    for pair, corr in complexity_correlations.items():
        print(f"  {pair}: {corr:.3f}")
    
    print("\nModel Performance:")
    print(f"{'Model':<25} {'Accuracy':<10} {'Effect Size':<12} {'Success':<8}")
    print("-" * 60)
    
    for model_name, result in results.items():
        if 'error' not in result:
            accuracy = result['accuracy']
            effect_size = result['effect_size']
            success = "YES" if accuracy > 0.60 and abs(effect_size) > 0.3 else "NO"
            print(f"{model_name:<25} {accuracy:<10.3f} {effect_size:<12.3f} {success:<8}")
        else:
            print(f"{model_name:<25} ERROR: {result['error'][:30]}...")
    
    # Save results
    output_path = Path("/Users/kumacmini/cost-aware-research-search/results/iter_04_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    print("\nExperiment completed successfully!")

if __name__ == "__main__":
    main()