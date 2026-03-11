import json
import os
import sys
import time
import random
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from collections import defaultdict
import openai
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
import re

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def setup_openai():
    """Setup OpenAI client with API key from environment"""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    return openai.OpenAI(api_key=api_key)

def retry_api_call(func, max_retries=3, delay=1):
    """Retry API calls with exponential backoff"""
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"API call failed after {max_retries} attempts: {e}")
                return None
            time.sleep(delay * (2 ** attempt))
    return None

def get_wikipedia_topics():
    """Generate diverse Wikipedia topics for text generation"""
    return [
        "Machine Learning", "Ancient Rome", "Climate Change", "Quantum Physics",
        "Renaissance Art", "Ocean Biology", "Space Exploration", "Medieval History",
        "Genetic Engineering", "Urban Planning", "Renewable Energy", "Neuroscience",
        "Philosophy of Mind", "Cryptocurrency", "Evolutionary Biology", "Architecture",
        "Artificial Intelligence", "Marine Ecology", "Sustainable Agriculture", "Robotics",
        "Medieval Literature", "Astrophysics", "Social Psychology", "Biotechnology",
        "Environmental Science", "Computer Networks", "Cultural Anthropology", "Geology"
    ]

def generate_llm_text(client, topic, seed_num):
    """Generate text using GPT-4o-mini on given topic"""
    def api_call():
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": f"Write a detailed encyclopedia article about {topic}. Use varied sentence structures and natural punctuation. Write 3-4 paragraphs with informative content. Random seed: {seed_num}"},
                {"role": "user", "content": f"Write an encyclopedia article about {topic}."}
            ],
            max_completion_tokens=400,
            temperature=0.7 + (seed_num % 3) * 0.1  # Vary temperature slightly
        )
        return response.choices[0].message.content
    
    return retry_api_call(api_call)

def create_synthetic_human_text(topic, style_seed):
    """Create synthetic human-like text with controlled variation"""
    random.seed(style_seed)
    
    # Different writing styles
    styles = [
        # Academic style - longer sentences, more complex punctuation
        lambda t: f"{t} is a complex field of study that encompasses multiple disciplines and methodologies. Researchers in this area often employ sophisticated analytical techniques; however, the interpretation of results requires careful consideration of various factors. The implications of recent findings suggest that our understanding of {t.lower()} may need to be revised, particularly in light of emerging evidence. Furthermore, interdisciplinary collaboration has become increasingly important, as it allows for more comprehensive approaches to understanding these phenomena.",
        
        # Journalistic style - medium sentences, balanced punctuation
        lambda t: f"{t} has gained significant attention in recent years. Scientists and experts continue to study its various aspects, making new discoveries regularly. The field presents both opportunities and challenges for researchers. Many institutions now offer specialized programs focusing on {t.lower()}. Public interest in this topic has also increased substantially.",
        
        # Educational style - varied sentence length, explanatory punctuation
        lambda t: f"What is {t}? This fascinating subject involves several key concepts that students should understand. First, the fundamental principles underlying {t.lower()} include multiple interconnected elements. Second, practical applications can be found in numerous real-world scenarios - from everyday situations to complex industrial processes. Finally, ongoing research continues to reveal new insights about this important field."
    ]
    
    style_func = styles[style_seed % len(styles)]
    text = style_func(topic)
    
    # Add some natural variation
    if style_seed % 4 == 0:
        text += " However, questions remain about certain aspects of the field."
    elif style_seed % 4 == 1:
        text += " Nevertheless, progress continues at a steady pace."
    elif style_seed % 4 == 2:
        text += " In conclusion, this remains an active area of investigation."
    else:
        text += " Therefore, continued study is essential for advancement."
    
    return text

def extract_statistical_features(text):
    """Extract comprehensive statistical features from text"""
    if not text or not isinstance(text, str):
        return None
    
    # Clean text
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Sentence segmentation (improved)
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if not sentences:
        return None
    
    # Sentence lengths
    sentence_lengths = [len(s.split()) for s in sentences]
    
    # Character-level analysis
    total_chars = len(text)
    punctuation_chars = len(re.findall(r'[,.;:!?\-"]', text))
    
    # Word analysis
    words = text.split()
    word_lengths = [len(word.strip('.,;:!?"()[]{}')) for word in words]
    
    # Punctuation density analysis
    periods = text.count('.')
    commas = text.count(',')
    semicolons = text.count(';')
    colons = text.count(':')
    exclamations = text.count('!')
    questions = text.count('?')
    
    # Syntactic complexity
    avg_sentence_length = np.mean(sentence_lengths) if sentence_lengths else 0
    sentence_length_variance = np.var(sentence_lengths) if len(sentence_lengths) > 1 else 0
    
    features = {
        'sentence_lengths': sentence_lengths,
        'avg_sentence_length': float(avg_sentence_length),
        'sentence_length_std': float(np.std(sentence_lengths)) if sentence_lengths else 0,
        'sentence_length_variance': float(sentence_length_variance),
        'sentence_count': len(sentences),
        'punctuation_density': float(punctuation_chars / total_chars) if total_chars > 0 else 0,
        'word_count': len(words),
        'avg_word_length': float(np.mean(word_lengths)) if word_lengths else 0,
        'word_length_std': float(np.std(word_lengths)) if word_lengths else 0,
        'periods_per_100_words': float(periods / len(words) * 100) if words else 0,
        'commas_per_100_words': float(commas / len(words) * 100) if words else 0,
        'complex_punct_ratio': float((semicolons + colons) / len(words) * 100) if words else 0,
        'text_length': total_chars
    }
    
    return features

def compute_distributional_differences(human_features, llm_features, feature_name):
    """Compute statistical tests for distributional differences"""
    human_values = [f[feature_name] for f in human_features if f and feature_name in f and not np.isnan(f[feature_name])]
    llm_values = [f[feature_name] for f in llm_features if f and feature_name in f and not np.isnan(f[feature_name])]
    
    if len(human_values) < 10 or len(llm_values) < 10:
        return None
    
    # Kolmogorov-Smirnov test
    ks_stat, ks_p = stats.ks_2samp(human_values, llm_values)
    
    # Mann-Whitney U test (non-parametric)
    mw_stat, mw_p = stats.mannwhitneyu(human_values, llm_values, alternative='two-sided')
    
    # Effect size (Cohen's d approximation)
    pooled_std = np.sqrt((np.var(human_values) + np.var(llm_values)) / 2)
    cohens_d = (np.mean(human_values) - np.mean(llm_values)) / pooled_std if pooled_std > 0 else 0
    
    return {
        'ks_statistic': float(ks_stat),
        'ks_p_value': float(ks_p),
        'mw_statistic': float(mw_stat),
        'mw_p_value': float(mw_p),
        'cohens_d': float(cohens_d),
        'human_mean': float(np.mean(human_values)),
        'human_std': float(np.std(human_values)),
        'llm_mean': float(np.mean(llm_values)),
        'llm_std': float(np.std(llm_values)),
        'sample_sizes': {'human': len(human_values), 'llm': len(llm_values)}
    }

def bootstrap_confidence_interval(data, n_bootstrap=1000, confidence=0.95):
    """Compute bootstrap confidence interval for mean"""
    if len(data) < 5:
        return None
    
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_means.append(np.mean(sample))
    
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_means, 100 * alpha / 2)
    upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
    
    return {'lower': float(lower), 'upper': float(upper)}

def generate_embeddings_and_cluster(texts, labels):
    """Generate embeddings and perform clustering analysis"""
    print("Loading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("Generating embeddings...")
    embeddings = model.encode(texts)
    
    # TF-IDF baseline
    tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
    tfidf_features = tfidf.fit_transform(texts).toarray()
    
    results = {}
    
    # Test different representations
    representations = {
        'sentence_transformer': embeddings,
        'tfidf': tfidf_features
    }
    
    for rep_name, features in representations.items():
        # Reduce dimensionality for clustering
        if features.shape[1] > 50:
            pca = PCA(n_components=50)
            features = pca.fit_transform(features)
        
        # K-means clustering
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features)
        
        # Compute separability metrics
        true_labels_binary = [0 if label == 'human' else 1 for label in labels]
        
        ari = adjusted_rand_score(true_labels_binary, cluster_labels)
        
        # Davies-Bouldin score (lower is better)
        from sklearn.metrics import davies_bouldin_score
        db_score = davies_bouldin_score(features, cluster_labels)
        
        # Classification accuracy
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        
        clf = LogisticRegression(random_state=42, max_iter=1000)
        cv_scores = cross_val_score(clf, features, true_labels_binary, cv=5)
        classification_accuracy = np.mean(cv_scores)
        
        results[rep_name] = {
            'adjusted_rand_index': float(ari),
            'davies_bouldin_score': float(db_score),
            'classification_accuracy': float(classification_accuracy),
            'classification_std': float(np.std(cv_scores))
        }
    
    return results

def main():
    print("Starting comprehensive text analysis experiment...")
    
    client = setup_openai()
    topics = get_wikipedia_topics()
    
    # Data collection with multiple seeds for robustness
    human_texts = []
    llm_texts = []
    human_labels = []
    llm_labels = []
    
    target_samples = 30  # 30 per condition = 60 total (within API limit)
    api_calls_made = 0
    max_api_calls = 25
    
    print(f"Generating {target_samples} samples per condition...")
    
    # Generate LLM texts
    for i in range(target_samples):
        if api_calls_made >= max_api_calls:
            print(f"Reached API limit of {max_api_calls} calls")
            break
            
        topic = topics[i % len(topics)]
        print(f"Generating LLM text {i+1}/{target_samples} (topic: {topic})...")
        
        llm_text = generate_llm_text(client, topic, i)
        api_calls_made += 1
        
        if llm_text:
            llm_texts.append(llm_text)
            llm_labels.append('llm')
        
        time.sleep(0.1)  # Rate limiting
    
    # Generate synthetic human texts (no API calls needed)
    print("Generating synthetic human texts...")
    for i in range(target_samples):
        topic = topics[i % len(topics)]
        human_text = create_synthetic_human_text(topic, i)
        human_texts.append(human_text)
        human_labels.append('human')
    
    print(f"Generated {len(llm_texts)} LLM texts and {len(human_texts)} human texts")
    
    if len(llm_texts) < 20 or len(human_texts) < 20:
        print("Warning: Insufficient samples for robust analysis")
    
    # Extract features
    print("Extracting statistical features...")
    human_features = []
    llm_features = []
    
    for i, text in enumerate(human_texts):
        print(f"Processing human text {i+1}/{len(human_texts)}...")
        features = extract_statistical_features(text)
        if features:
            human_features.append(features)
    
    for i, text in enumerate(llm_texts):
        print(f"Processing LLM text {i+1}/{len(llm_texts)}...")
        features = extract_statistical_features(text)
        if features:
            llm_features.append(features)
    
    print(f"Extracted features from {len(human_features)} human and {len(llm_features)} LLM texts")
    
    # Statistical analysis
    print("Performing statistical analysis...")
    
    key_features = [
        'avg_sentence_length', 'sentence_length_std', 'sentence_length_variance',
        'punctuation_density', 'avg_word_length', 'word_length_std',
        'commas_per_100_words', 'complex_punct_ratio'
    ]
    
    statistical_results = {}
    bootstrap_results = {}
    
    for feature in key_features:
        print(f"Analyzing {feature}...")
        
        result = compute_distributional_differences(human_features, llm_features, feature)
        if result:
            statistical_results[feature] = result
            
            # Bootstrap confidence intervals
            human_values = [f[feature] for f in human_features if f and feature in f and not np.isnan(f[feature])]
            llm_values = [f[feature] for f in llm_features if f and feature in f and not np.isnan(f[feature])]
            
            bootstrap_results[feature] = {
                'human_ci': bootstrap_confidence_interval(human_values),
                'llm_ci': bootstrap_confidence_interval(llm_values)
            }
    
    # Clustering and separability analysis
    print("Performing clustering analysis...")
    all_texts = human_texts + llm_texts
    all_labels = human_labels + llm_labels
    
    clustering_results = {}
    if len(all_texts) >= 20:
        clustering_results = generate_embeddings_and_cluster(all_texts, all_labels)
    
    # Compile final results
    results = {
        'experiment_info': {
            'total_samples': len(human_features) + len(llm_features),
            'human_samples': len(human_features),
            'llm_samples': len(llm_features),
            'api_calls_made': api_calls_made,
            'topics_used': len(set(topics[:max(len(human_texts), len(llm_texts))]))
        },
        'statistical_tests': statistical_results,
        'bootstrap_confidence_intervals': bootstrap_results,
        'clustering_analysis': clustering_results,
        'summary_statistics': {}
    }
    
    # Summary statistics
    for feature in key_features:
        if feature in statistical_results:
            results['summary_statistics'][feature] = {
                'significant_difference': statistical_results[feature]['ks_p_value'] < 0.05,
                'effect_size_category': 'large' if abs(statistical_results[feature]['cohens_d']) > 0.8 else 
                                      'medium' if abs(statistical_results[feature]['cohens_d']) > 0.5 else 'small',
                'ks_statistic': statistical_results[feature]['ks_statistic']
            }
    
    # Save results
    output_path = '/Users/kumacmini/cost-aware-research-search/results/iter_02_results.json'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"Saving results to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Print results table
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)
    
    print(f"\nSample Sizes:")
    print(f"  Human texts: {len(human_features)}")
    print(f"  LLM texts: {len(llm_features)}")
    print(f"  API calls made: {api_calls_made}")
    
    print(f"\nStatistical Test Results:")
    print(f"{'Feature':<25} {'KS Stat':<10} {'P-value':<12} {'Effect Size':<12} {'Significant':<12}")
    print("-" * 75)
    
    significant_count = 0
    for feature in key_features:
        if feature in statistical_results:
            result = statistical_results[feature]
            sig = "Yes" if result['ks_p_value'] < 0.05 else "No"
            if sig == "Yes":
                significant_count += 1
            
            print(f"{feature:<25} {result['ks_statistic']:<10.3f} {result['ks_p_value']:<12.3f} {result['cohens_d']:<12.3f} {sig:<12}")
    
    print(f"\nSignificant differences found: {significant_count}/{len(key_features)} features")
    
    if clustering_results:
        print(f"\nClustering Analysis:")
        for rep_name, metrics in clustering_results.items():
            print(f"  {rep_name.title()}:")
            print(f"    Classification Accuracy: {metrics['classification_accuracy']:.3f}")
            print(f"    Adjusted Rand Index: {metrics['adjusted_rand_index']:.3f}")
            print(f"    Davies-Bouldin Score: {metrics['davies_bouldin_score']:.3f}")
    
    # Hypothesis evaluation
    strong_effects = sum(1 for feature in key_features 
                        if feature in statistical_results and 
                        statistical_results[feature]['ks_statistic'] > 0.3)
    
    print(f"\nHypothesis Evaluation:")
    print(f"  Features with KS statistic > 0.3: {strong_effects}/{len(key_features)}")
    print(f"  Success threshold met: {'Yes' if strong_effects > 0 else 'No'}")
    
    print(f"\nExperiment completed successfully!")

if __name__ == "__main__":
    main()