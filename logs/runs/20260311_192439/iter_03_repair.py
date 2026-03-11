import os
import json
import numpy as np
import time
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import stats
import matplotlib.pyplot as plt

def create_benchmark_pairs():
    """Create a robust set of similar and dissimilar sentence pairs"""
    
    # High similarity pairs (paraphrases and near-synonyms)
    similar_pairs = [
        ("The cat is sleeping on the couch", "A cat is resting on the sofa"),
        ("She drives to work every morning", "Every morning, she commutes to work by car"),
        ("The movie was really entertaining", "The film was very enjoyable"),
        ("It's raining heavily outside", "There's a heavy downpour outdoors"),
        ("The book contains interesting stories", "This book has fascinating tales"),
        ("He enjoys playing basketball", "Basketball is his favorite sport to play"),
        ("The restaurant serves delicious food", "This eatery offers tasty meals"),
        ("She studies mathematics at university", "Mathematics is her major in college"),
        ("The dog is barking loudly", "A dog is making loud barking sounds"),
        ("They went shopping for groceries", "They purchased food items at the store"),
        ("The weather is sunny today", "It's a bright, sunny day"),
        ("He works as a software engineer", "His profession is software development"),
        ("The children are playing in the park", "Kids are having fun at the playground"),
        ("She loves reading mystery novels", "Mystery books are her favorite genre"),
        ("The train arrives at 3 PM", "The train is scheduled to arrive at three o'clock"),
        ("He cooks dinner every night", "Every evening, he prepares the dinner meal"),
        ("The building is very tall", "This structure has significant height"),
        ("She speaks three languages fluently", "She is fluent in three different languages"),
        ("The concert was absolutely amazing", "The musical performance was incredible"),
        ("They traveled to Europe last summer", "Last summer, they took a European vacation"),
        ("The computer is running slowly", "This computer has poor performance speed"),
        ("She teaches elementary school students", "Her job is teaching young children"),
        ("The garden has beautiful flowers", "Beautiful blooms fill the garden space"),
        ("He exercises at the gym regularly", "Regular gym workouts are part of his routine"),
        ("The meeting starts at noon", "The conference begins at twelve o'clock"),
        ("She bought a new dress yesterday", "Yesterday, she purchased a new outfit"),
        ("The coffee tastes bitter", "This coffee has a bitter flavor"),
        ("They live in a small apartment", "Their residence is a compact flat"),
        ("The sunset looks magnificent", "The evening sky appears spectacular"),
        ("He studies hard for exams", "Exam preparation requires his serious effort")
    ]
    
    # Low similarity pairs (different topics and contexts)
    dissimilar_pairs = [
        ("The cat is sleeping on the couch", "Quantum physics explains particle behavior"),
        ("She drives to work every morning", "The ancient pyramids were built in Egypt"),
        ("The movie was really entertaining", "Photosynthesis occurs in plant leaves"),
        ("It's raining heavily outside", "The stock market closed higher today"),
        ("The book contains interesting stories", "Basketball players need good coordination"),
        ("He enjoys playing basketball", "The recipe calls for two cups of flour"),
        ("The restaurant serves delicious food", "Democracy evolved over many centuries"),
        ("She studies mathematics at university", "The ocean contains diverse marine life"),
        ("The dog is barking loudly", "Computer algorithms process data efficiently"),
        ("They went shopping for groceries", "Renaissance art influenced modern painters"),
        ("The weather is sunny today", "Mitochondria generate cellular energy"),
        ("He works as a software engineer", "The Great Wall spans thousands of miles"),
        ("The children are playing in the park", "Chemical reactions follow thermodynamic laws"),
        ("She loves reading mystery novels", "Satellites orbit Earth at precise altitudes"),
        ("The train arrives at 3 PM", "Beethoven composed nine symphonies"),
        ("He cooks dinner every night", "Volcanic eruptions reshape geological landscapes"),
        ("The building is very tall", "DNA contains genetic information"),
        ("She speaks three languages fluently", "Black holes bend spacetime significantly"),
        ("The concert was absolutely amazing", "Ecosystems maintain delicate natural balances"),
        ("They traveled to Europe last summer", "Artificial intelligence processes complex patterns"),
        ("The computer is running slowly", "The Constitution establishes governmental framework"),
        ("She teaches elementary school students", "Glaciers carved ancient valley systems"),
        ("The garden has beautiful flowers", "Economic theories explain market behaviors"),
        ("He exercises at the gym regularly", "Archaeological discoveries reveal historical cultures"),
        ("The meeting starts at noon", "Electromagnetic waves travel at light speed"),
        ("She bought a new dress yesterday", "Philosophy examines fundamental existence questions"),
        ("The coffee tastes bitter", "Tectonic plates cause continental drift"),
        ("They live in a small apartment", "Medical research advances treatment options"),
        ("The sunset looks magnificent", "Mathematical proofs establish logical certainty"),
        ("He studies hard for exams", "Renewable energy sources reduce carbon emissions")
    ]
    
    return similar_pairs, dissimilar_pairs

def compute_jaccard_similarity(text1, text2):
    """Compute Jaccard similarity as lexical baseline"""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    return len(intersection) / len(union) if union else 0

def compute_tfidf_similarity(pairs):
    """Compute TF-IDF cosine similarity baseline"""
    similarities = []
    vectorizer = TfidfVectorizer()
    
    for text1, text2 in pairs:
        tfidf_matrix = vectorizer.fit_transform([text1, text2])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        similarities.append(similarity)
    
    return similarities

def bootstrap_confidence_interval(data, n_bootstrap=1000, confidence_level=0.95):
    """Compute bootstrap confidence interval for mean"""
    bootstrap_means = []
    n = len(data)
    
    for _ in range(n_bootstrap):
        bootstrap_sample = np.random.choice(data, size=n, replace=True)
        bootstrap_means.append(np.mean(bootstrap_sample))
    
    alpha = 1 - confidence_level
    lower_percentile = (alpha/2) * 100
    upper_percentile = (100 - alpha/2)
    
    ci_lower = np.percentile(bootstrap_means, lower_percentile)
    ci_upper = np.percentile(bootstrap_means, upper_percentile)
    
    return ci_lower, ci_upper

def compute_cohens_d(group1, group2):
    """Compute Cohen's d effect size"""
    mean1, mean2 = np.mean(group1), np.mean(group2)
    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    n1, n2 = len(group1), len(group2)
    
    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
    cohens_d = (mean1 - mean2) / pooled_std
    
    return cohens_d

def main():
    start_time = time.time()
    results = {"experiment_info": {}, "results": {}, "statistical_tests": {}, "diagnostics": {}}
    
    try:
        print("Starting Semantic Similarity Analysis with Statistical Validation...")
        
        # Create benchmark data
        similar_pairs, dissimilar_pairs = create_benchmark_pairs()
        all_pairs = similar_pairs + dissimilar_pairs
        labels = [1] * len(similar_pairs) + [0] * len(dissimilar_pairs)
        
        print(f"Dataset: {len(similar_pairs)} similar pairs, {len(dissimilar_pairs)} dissimilar pairs")
        
        # Load embedding model
        print("Loading sentence transformer model...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Compute sentence embeddings
        print("Computing sentence embeddings...")
        sentences1 = [pair[0] for pair in all_pairs]
        sentences2 = [pair[1] for pair in all_pairs]
        
        embeddings1 = model.encode(sentences1, show_progress_bar=False)
        embeddings2 = model.encode(sentences2, show_progress_bar=False)
        
        # Compute cosine similarities (not distances)
        cosine_similarities = []
        for i in range(len(all_pairs)):
            sim = cosine_similarity([embeddings1[i]], [embeddings2[i]])[0][0]
            cosine_similarities.append(sim)
        
        # Split similarities by labels
        similar_scores = [cosine_similarities[i] for i in range(len(labels)) if labels[i] == 1]
        dissimilar_scores = [cosine_similarities[i] for i in range(len(labels)) if labels[i] == 0]
        
        print(f"Similar pairs mean similarity: {np.mean(similar_scores):.4f}")
        print(f"Dissimilar pairs mean similarity: {np.mean(dissimilar_scores):.4f}")
        
        # Statistical tests
        print("\nPerforming statistical tests...")
        
        # Welch's t-test
        t_stat, p_value = stats.ttest_ind(similar_scores, dissimilar_scores, equal_var=False)
        cohens_d = compute_cohens_d(similar_scores, dissimilar_scores)
        
        # Bootstrap confidence intervals
        similar_ci = bootstrap_confidence_interval(similar_scores)
        dissimilar_ci = bootstrap_confidence_interval(dissimilar_scores)
        
        # Correlation with binary labels
        correlation, corr_p_value = stats.pearsonr(cosine_similarities, labels)
        
        print(f"T-test: t={t_stat:.4f}, p={p_value:.6f}")
        print(f"Cohen's d: {cohens_d:.4f}")
        print(f"Correlation with labels: r={correlation:.4f}, p={corr_p_value:.6f}")
        
        # Diagnostic baselines
        print("\nComputing diagnostic baselines...")
        
        # Random pairs baseline
        np.random.seed(42)
        random_indices = np.random.choice(len(all_pairs), size=20, replace=False)
        random_pairs = [all_pairs[i] for i in random_indices]
        random_embeddings1 = model.encode([pair[0] for pair in random_pairs], show_progress_bar=False)
        random_embeddings2 = model.encode([pair[1] for pair in random_pairs], show_progress_bar=False)
        random_similarities = [cosine_similarity([random_embeddings1[i]], [random_embeddings2[i]])[0][0] 
                             for i in range(len(random_pairs))]
        
        # Jaccard similarity baseline
        jaccard_similarities = [compute_jaccard_similarity(pair[0], pair[1]) for pair in all_pairs]
        jaccard_similar = [jaccard_similarities[i] for i in range(len(labels)) if labels[i] == 1]
        jaccard_dissimilar = [jaccard_similarities[i] for i in range(len(labels)) if labels[i] == 0]
        
        # TF-IDF similarity baseline
        tfidf_similarities = compute_tfidf_similarity(all_pairs)
        tfidf_similar = [tfidf_similarities[i] for i in range(len(labels)) if labels[i] == 1]
        tfidf_dissimilar = [tfidf_similarities[i] for i in range(len(labels)) if labels[i] == 0]
        
        # Store results with proper JSON serialization
        results["experiment_info"] = {
            "model_name": "all-MiniLM-L6-v2",
            "n_similar_pairs": len(similar_pairs),
            "n_dissimilar_pairs": len(dissimilar_pairs),
            "total_pairs": len(all_pairs),
            "timestamp": datetime.now().isoformat()
        }
        
        results["results"] = {
            "embedding_similarities": {
                "similar_pairs_mean": float(np.mean(similar_scores)),
                "similar_pairs_std": float(np.std(similar_scores)),
                "dissimilar_pairs_mean": float(np.mean(dissimilar_scores)),
                "dissimilar_pairs_std": float(np.std(dissimilar_scores)),
                "all_scores": [float(x) for x in cosine_similarities],
                "labels": labels
            }
        }
        
        results["statistical_tests"] = {
            "welch_t_test": {
                "t_statistic": float(t_stat),
                "p_value": float(p_value),
                "significant": bool(p_value < 0.05)
            },
            "effect_size": {
                "cohens_d": float(cohens_d),
                "large_effect": bool(abs(cohens_d) > 0.5)
            },
            "bootstrap_ci": {
                "similar_pairs_ci": [float(similar_ci[0]), float(similar_ci[1])],
                "dissimilar_pairs_ci": [float(dissimilar_ci[0]), float(dissimilar_ci[1])]
            },
            "correlation": {
                "pearson_r": float(correlation),
                "p_value": float(corr_p_value)
            }
        }
        
        results["diagnostics"] = {
            "random_baseline": {
                "mean_similarity": float(np.mean(random_similarities)),
                "std_similarity": float(np.std(random_similarities))
            },
            "jaccard_baseline": {
                "similar_pairs_mean": float(np.mean(jaccard_similar)),
                "dissimilar_pairs_mean": float(np.mean(jaccard_dissimilar)),
                "t_test_p": float(stats.ttest_ind(jaccard_similar, jaccard_dissimilar)[1])
            },
            "tfidf_baseline": {
                "similar_pairs_mean": float(np.mean(tfidf_similar)),
                "dissimilar_pairs_mean": float(np.mean(tfidf_dissimilar)),
                "t_test_p": float(stats.ttest_ind(tfidf_similar, tfidf_dissimilar)[1])
            }
        }
        
        # Summary evaluation
        hypothesis_supported = (
            p_value < 0.05 and 
            abs(cohens_d) > 0.5 and 
            np.mean(similar_scores) > np.mean(dissimilar_scores)
        )
        
        results["hypothesis_evaluation"] = {
            "supported": bool(hypothesis_supported),
            "criteria_met": {
                "statistical_significance": bool(p_value < 0.05),
                "large_effect_size": bool(abs(cohens_d) > 0.5),
                "correct_direction": bool(np.mean(similar_scores) > np.mean(dissimilar_scores))
            }
        }
        
        # Print summary
        print("\n" + "="*60)
        print("EXPERIMENT SUMMARY")
        print("="*60)
        print(f"Sample Size: {len(all_pairs)} pairs ({len(similar_pairs)} similar, {len(dissimilar_pairs)} dissimilar)")
        print(f"Embedding Model: all-MiniLM-L6-v2")
        print(f"Similar pairs similarity: {np.mean(similar_scores):.4f} ± {np.std(similar_scores):.4f}")
        print(f"Dissimilar pairs similarity: {np.mean(dissimilar_scores):.4f} ± {np.std(dissimilar_scores):.4f}")
        print(f"Statistical significance: p = {p_value:.6f} ({'✓' if p_value < 0.05 else '✗'})")
        print(f"Effect size (Cohen's d): {cohens_d:.4f} ({'✓' if abs(cohens_d) > 0.5 else '✗'})")
        print(f"Hypothesis supported: {'YES' if hypothesis_supported else 'NO'}")
        print("="*60)
        
        # Save results
        output_path = "/Users/kumacmini/cost-aware-research-search/results/iter_03_results.json"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")
        print(f"Total runtime: {time.time() - start_time:.2f} seconds")
        
        return results
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        results["error"] = str(e)
        
        # Still try to save partial results
        try:
            output_path = "/Users/kumacmini/cost-aware-research-search/results/iter_03_results.json"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
        except:
            pass
        
        return results

if __name__ == "__main__":
    main()