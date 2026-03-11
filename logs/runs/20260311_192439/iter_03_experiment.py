import os
import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings('ignore')

def bootstrap_confidence_interval(data, n_bootstrap=1000, confidence_level=0.95):
    """Calculate bootstrap confidence interval for mean."""
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_means.append(np.mean(sample))
    
    alpha = 1 - confidence_level
    lower = np.percentile(bootstrap_means, 100 * (alpha/2))
    upper = np.percentile(bootstrap_means, 100 * (1 - alpha/2))
    return lower, upper

def jaccard_similarity(text1, text2):
    """Calculate Jaccard similarity between two texts."""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    return len(intersection) / len(union) if len(union) > 0 else 0

def cohens_d(group1, group2):
    """Calculate Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    pooled_std = np.sqrt(((n1-1)*np.var(group1, ddof=1) + (n2-1)*np.var(group2, ddof=1)) / (n1+n2-2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std > 0 else 0

def create_sts_like_pairs():
    """Create STS-B like sentence pairs with verified similarity labels."""
    # High similarity pairs (paraphrases and near-synonymous sentences)
    high_similarity_pairs = [
        ("A person is riding a bicycle", "Someone is cycling on a bike"),
        ("The cat is sleeping on the couch", "A feline is resting on the sofa"),
        ("It's raining heavily outside", "There's a heavy downpour outdoors"),
        ("The student studied hard for the exam", "The pupil prepared intensively for the test"),
        ("The company announced new hiring plans", "The firm revealed fresh recruitment strategies"),
        ("The book was very interesting to read", "The novel was quite engaging and captivating"),
        ("She cooked dinner for her family", "She prepared a meal for her relatives"),
        ("The car broke down on the highway", "The vehicle malfunctioned on the freeway"),
        ("They went shopping at the mall", "They visited the shopping center to buy things"),
        ("The weather is beautiful today", "It's a lovely day with great weather"),
        ("He plays guitar in a band", "He's a guitarist in a musical group"),
        ("The meeting was scheduled for tomorrow", "The conference is planned for the next day"),
        ("She loves watching movies at home", "She enjoys viewing films in her house"),
        ("The dog barked loudly at strangers", "The canine made loud noises at unfamiliar people"),
        ("Children are playing in the park", "Kids are having fun at the playground"),
        ("The restaurant serves delicious food", "The eatery offers tasty cuisine"),
        ("He drives to work every morning", "He commutes to his job by car daily"),
        ("The concert was sold out quickly", "The musical performance tickets sold rapidly"),
        ("She graduated from university last year", "She completed her degree program twelve months ago"),
        ("The phone battery died completely", "The mobile device power was fully depleted"),
        ("They moved to a new apartment", "They relocated to a different residence"),
        ("The teacher explained the lesson clearly", "The instructor described the topic with clarity"),
        ("He bought groceries at the store", "He purchased food items at the shop"),
        ("The flight was delayed by two hours", "The airplane departure was postponed for 120 minutes"),
        ("She works as a software engineer", "She's employed as a computer programmer"),
        ("The garden has beautiful flowers", "The yard contains lovely blossoms"),
        ("They celebrated their anniversary yesterday", "They commemorated their special day the previous day"),
    ]
    
    # Low similarity pairs (topically different or contradictory)
    low_similarity_pairs = [
        ("A person is riding a bicycle", "The stock market crashed today"),
        ("The cat is sleeping on the couch", "Scientists discovered a new planet"),
        ("It's raining heavily outside", "She baked chocolate cookies for dessert"),
        ("The student studied hard for the exam", "The ocean waves were massive yesterday"),
        ("The company announced new hiring plans", "My grandmother knits beautiful sweaters"),
        ("The book was very interesting to read", "The mountain climber reached the summit"),
        ("She cooked dinner for her family", "The football team won the championship"),
        ("The car broke down on the highway", "Dolphins are intelligent marine mammals"),
        ("They went shopping at the mall", "The ancient ruins were recently excavated"),
        ("The weather is beautiful today", "He solved complex mathematical equations"),
        ("He plays guitar in a band", "The farmer harvested corn this season"),
        ("The meeting was scheduled for tomorrow", "She painted abstract art in her studio"),
        ("She loves watching movies at home", "The volcano erupted without warning"),
        ("The dog barked loudly at strangers", "They invented a revolutionary new technology"),
        ("Children are playing in the park", "The historian researched medieval manuscripts"),
        ("The restaurant serves delicious food", "The astronaut performed spacewalks successfully"),
        ("He drives to work every morning", "The butterfly migration lasted several weeks"),
        ("The concert was sold out quickly", "She studies marine biology extensively"),
        ("She graduated from university last year", "The detective solved the mysterious case"),
        ("The phone battery died completely", "They discovered oil reserves underground"),
        ("They moved to a new apartment", "The chess master won the tournament"),
        ("The teacher explained the lesson clearly", "The glacier melted at an alarming rate"),
        ("He bought groceries at the store", "She designed innovative architectural structures"),
        ("The flight was delayed by two hours", "The poet wrote verses about nature"),
        ("She works as a software engineer", "The circus performer amazed the audience"),
        ("The garden has beautiful flowers", "He studies quantum physics theories"),
        ("They celebrated their anniversary yesterday", "The submarine explored deep ocean trenches"),
    ]
    
    return high_similarity_pairs, low_similarity_pairs

def create_random_pairs(sentences, n_pairs=27):
    """Create random sentence pairs from available sentences."""
    all_sentences = []
    high_sim, low_sim = create_sts_like_pairs()
    for pair in high_sim + low_sim:
        all_sentences.extend(pair)
    
    random_pairs = []
    for _ in range(n_pairs):
        sent1, sent2 = np.random.choice(all_sentences, 2, replace=False)
        random_pairs.append((sent1, sent2))
    
    return random_pairs

def main():
    print("Starting Semantic Similarity Analysis with Statistical Testing")
    print("=" * 60)
    
    # Initialize results structure
    results = {
        'experiment_metadata': {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'hypothesis': 'Sentence embeddings show significantly higher similarity for semantically similar pairs',
            'sample_sizes': {},
            'models_tested': []
        },
        'raw_data': {
            'similar_pairs_scores': [],
            'dissimilar_pairs_scores': [],
            'random_pairs_scores': [],
            'pair_level_data': []
        },
        'statistical_tests': {},
        'baselines': {},
        'diagnostics': {},
        'summary_statistics': {}
    }
    
    try:
        # Create datasets
        print("Creating sentence pair datasets...")
        high_similarity_pairs, low_similarity_pairs = create_sts_like_pairs()
        random_pairs = create_random_pairs([], n_pairs=27)
        
        # Verify minimum sample size
        total_samples = len(high_similarity_pairs) + len(low_similarity_pairs) + len(random_pairs)
        print(f"Total sample size: {total_samples}")
        assert total_samples >= 50, f"Insufficient samples: {total_samples} < 50"
        
        results['experiment_metadata']['sample_sizes'] = {
            'similar_pairs': len(high_similarity_pairs),
            'dissimilar_pairs': len(low_similarity_pairs),
            'random_pairs': len(random_pairs),
            'total': total_samples
        }
        
        # Load sentence transformer model
        print("Loading sentence transformer model...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        results['experiment_metadata']['models_tested'].append('all-MiniLM-L6-v2')
        
        # Process all pairs
        all_pairs = [
            (high_similarity_pairs, 'similar'),
            (low_similarity_pairs, 'dissimilar'), 
            (random_pairs, 'random')
        ]
        
        similarity_scores = {'similar': [], 'dissimilar': [], 'random': []}
        jaccard_scores = {'similar': [], 'dissimilar': [], 'random': []}
        tfidf_scores = {'similar': [], 'dissimilar': [], 'random': []}
        
        # Initialize TF-IDF vectorizer for baseline
        tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        all_sentences = []
        for pairs, label in all_pairs:
            for sent1, sent2 in pairs:
                all_sentences.extend([sent1, sent2])
        tfidf_vectorizer.fit(all_sentences)
        
        print("Computing embeddings and similarities...")
        
        for pairs, label in all_pairs:
            print(f"Processing {len(pairs)} {label} pairs...")
            
            for sent1, sent2 in pairs:
                # Sentence transformer embeddings
                embeddings = model.encode([sent1, sent2])
                # Use cosine similarity (not distance)
                cos_sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
                similarity_scores[label].append(cos_sim)
                
                # Jaccard similarity baseline
                jaccard_sim = jaccard_similarity(sent1, sent2)
                jaccard_scores[label].append(jaccard_sim)
                
                # TF-IDF cosine similarity baseline
                tfidf_vecs = tfidf_vectorizer.transform([sent1, sent2])
                tfidf_sim = cosine_similarity(tfidf_vecs[0], tfidf_vecs[1])[0][0]
                tfidf_scores[label].append(tfidf_sim)
                
                # Store pair-level data
                pair_data = {
                    'sentence1': sent1,
                    'sentence2': sent2,
                    'true_label': label,
                    'cosine_similarity': float(cos_sim),
                    'jaccard_similarity': float(jaccard_sim),
                    'tfidf_similarity': float(tfidf_sim)
                }
                results['raw_data']['pair_level_data'].append(pair_data)
        
        # Store raw scores
        results['raw_data']['similar_pairs_scores'] = [float(x) for x in similarity_scores['similar']]
        results['raw_data']['dissimilar_pairs_scores'] = [float(x) for x in similarity_scores['dissimilar']]
        results['raw_data']['random_pairs_scores'] = [float(x) for x in similarity_scores['random']]
        
        # Statistical analysis
        print("\nPerforming statistical tests...")
        
        # Welch's t-test (similar vs dissimilar)
        t_stat, p_value = stats.ttest_ind(
            similarity_scores['similar'], 
            similarity_scores['dissimilar'], 
            equal_var=False
        )
        
        # Effect size (Cohen's d)
        effect_size = cohens_d(similarity_scores['similar'], similarity_scores['dissimilar'])
        
        # Bootstrap confidence intervals
        similar_ci = bootstrap_confidence_interval(similarity_scores['similar'])
        dissimilar_ci = bootstrap_confidence_interval(similarity_scores['dissimilar'])
        
        results['statistical_tests'] = {
            'welch_t_test': {
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'significant': p_value < 0.05
            },
            'effect_size_cohens_d': float(effect_size),
            'bootstrap_confidence_intervals': {
                'similar_pairs_95ci': [float(similar_ci[0]), float(similar_ci[1])],
                'dissimilar_pairs_95ci': [float(dissimilar_ci[0]), float(dissimilar_ci[1])]
            }
        }
        
        # Summary statistics
        results['summary_statistics'] = {
            'similar_pairs': {
                'mean': float(np.mean(similarity_scores['similar'])),
                'std': float(np.std(similarity_scores['similar'])),
                'median': float(np.median(similarity_scores['similar'])),
                'min': float(np.min(similarity_scores['similar'])),
                'max': float(np.max(similarity_scores['similar']))
            },
            'dissimilar_pairs': {
                'mean': float(np.mean(similarity_scores['dissimilar'])),
                'std': float(np.std(similarity_scores['dissimilar'])),
                'median': float(np.median(similarity_scores['dissimilar'])),
                'min': float(np.min(similarity_scores['dissimilar'])),
                'max': float(np.max(similarity_scores['dissimilar']))
            },
            'random_pairs': {
                'mean': float(np.mean(similarity_scores['random'])),
                'std': float(np.std(similarity_scores['random'])),
                'median': float(np.median(similarity_scores['random'])),
                'min': float(np.min(similarity_scores['random'])),
                'max': float(np.max(similarity_scores['random']))
            }
        }
        
        # Baseline comparisons
        results['baselines'] = {
            'jaccard_similarity': {
                'similar_mean': float(np.mean(jaccard_scores['similar'])),
                'dissimilar_mean': float(np.mean(jaccard_scores['dissimilar'])),
                'random_mean': float(np.mean(jaccard_scores['random']))
            },
            'tfidf_similarity': {
                'similar_mean': float(np.mean(tfidf_scores['similar'])),
                'dissimilar_mean': float(np.mean(tfidf_scores['dissimilar'])),
                'random_mean': float(np.mean(tfidf_scores['random']))
            }
        }
        
        # Correlation analysis
        human_labels = []
        model_scores = []
        for pair_data in results['raw_data']['pair_level_data']:
            if pair_data['true_label'] == 'similar':
                human_labels.append(1)
            else:
                human_labels.append(0)
            model_scores.append(pair_data['cosine_similarity'])
        
        correlation, corr_p_value = stats.pearsonr(human_labels, model_scores)
        results['statistical_tests']['correlation_analysis'] = {
            'pearson_r': float(correlation),
            'p_value': float(corr_p_value),
            'significant': corr_p_value < 0.05
        }
        
        # Print results
        print("\n" + "="*60)
        print("RESULTS SUMMARY")
        print("="*60)
        
        print(f"\nSample Sizes:")
        print(f"  Similar pairs: {len(similarity_scores['similar'])}")
        print(f"  Dissimilar pairs: {len(similarity_scores['dissimilar'])}")
        print(f"  Random pairs: {len(similarity_scores['random'])}")
        print(f"  Total: {total_samples}")
        
        print(f"\nMean Cosine Similarity Scores:")
        print(f"  Similar pairs: {np.mean(similarity_scores['similar']):.4f} ± {np.std(similarity_scores['similar']):.4f}")
        print(f"  Dissimilar pairs: {np.mean(similarity_scores['dissimilar']):.4f} ± {np.std(similarity_scores['dissimilar']):.4f}")
        print(f"  Random pairs: {np.mean(similarity_scores['random']):.4f} ± {np.std(similarity_scores['random']):.4f}")
        
        print(f"\nStatistical Tests:")
        print(f"  Welch's t-test (similar vs dissimilar):")
        print(f"    t = {t_stat:.4f}, p = {p_value:.6f}")
        print(f"    Significant: {'Yes' if p_value < 0.05 else 'No'}")
        print(f"  Effect size (Cohen's d): {effect_size:.4f}")
        print(f"  Correlation with labels: r = {correlation:.4f}, p = {corr_p_value:.6f}")
        
        print(f"\nBaseline Comparisons:")
        print(f"  Jaccard similarity - Similar: {np.mean(jaccard_scores['similar']):.4f}, Dissimilar: {np.mean(jaccard_scores['dissimilar']):.4f}")
        print(f"  TF-IDF similarity - Similar: {np.mean(tfidf_scores['similar']):.4f}, Dissimilar: {np.mean(tfidf_scores['dissimilar']):.4f}")
        
        success_criteria = {
            'effect_size_sufficient': effect_size > 0.5,
            'statistically_significant': p_value < 0.05,
            'proper_direction': np.mean(similarity_scores['similar']) > np.mean(similarity_scores['dissimilar'])
        }
        
        print(f"\nSuccess Criteria:")
        for criterion, met in success_criteria.items():
            print(f"  {criterion}: {'✓' if met else '✗'}")
        
        results['success_criteria'] = {k: bool(v) for k, v in success_criteria.items()}
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.boxplot([similarity_scores['similar'], similarity_scores['dissimilar'], similarity_scores['random']], 
                   labels=['Similar', 'Dissimilar', 'Random'])
        plt.title('Cosine Similarity Distributions')
        plt.ylabel('Cosine Similarity')
        
        plt.subplot(2, 2, 2)
        plt.hist(similarity_scores['similar'], alpha=0.5, label='Similar', bins=15)
        plt.hist(similarity_scores['dissimilar'], alpha=0.5, label='Dissimilar', bins=15)
        plt.hist(similarity_scores['random'], alpha=0.5, label='Random', bins=15)
        plt.xlabel('Cosine Similarity')
        plt.ylabel('Frequency')
        plt.title('Similarity Score Distributions')
        plt.legend()
        
        plt.subplot(2, 2, 3)
        plt.scatter(human_labels, model_scores, alpha=0.6)
        plt.xlabel('Human Label (0=Dissimilar, 1=Similar)')
        plt.ylabel('Model Cosine Similarity')
        plt.title(f'Model vs Human Labels (r={correlation:.3f})')
        
        plt.subplot(2, 2, 4)
        methods = ['Sentence\nTransformer', 'TF-IDF', 'Jaccard']
        similar_means = [
            np.mean(similarity_scores['similar']),
            np.mean(tfidf_scores['similar']),
            np.mean(jaccard_scores['similar'])
        ]
        dissimilar_means = [
            np.mean(similarity_scores['dissimilar']),
            np.mean(tfidf_scores['dissimilar']),
            np.mean(jaccard_scores['dissimilar'])
        ]
        
        x = np.arange(len(methods))
        width = 0.35
        plt.bar(x - width/2, similar_means, width, label='Similar', alpha=0.8)
        plt.bar(x + width/2, dissimilar_means, width, label='Dissimilar', alpha=0.8)
        plt.xlabel('Method')
        plt.ylabel('Mean Similarity')
        plt.title('Baseline Comparison')
        plt.xticks(x, methods)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('/Users/kumacmini/cost-aware-research-search/results/iter_03_similarity_analysis.png', 
                   dpi=150, bbox_inches='tight')
        plt.show()
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        results['error'] = str(e)
        import traceback
        results['traceback'] = traceback.format_exc()
    
    finally:
        # Save results to JSON
        output_path = '/Users/kumacmini/cost-aware-research-search/results/iter_03_results.json'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        try:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\nResults saved to: {output_path}")
        except Exception as e:
            print(f"Error saving results: {str(e)}")
    
    print("\nExperiment completed!")

if __name__ == "__main__":
    main()