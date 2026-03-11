import os
import json
import numpy as np
import nltk
from nltk.corpus import words
from nltk import word_tokenize, pos_tag
import random
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')
try:
    nltk.data.find('corpora/words')
except LookupError:
    nltk.download('words')

def calculate_syntactic_complexity(sentence):
    """Calculate syntactic complexity using multiple metrics"""
    tokens = word_tokenize(sentence)
    pos_tags = pos_tag(tokens)
    
    # Number of words
    word_count = len(tokens)
    
    # Average word length
    avg_word_length = np.mean([len(word) for word in tokens if word.isalpha()])
    
    # Number of subordinating conjunctions and relative pronouns (complexity indicators)
    complex_pos = ['IN', 'WDT', 'WP', 'WRB']
    complex_count = sum(1 for _, pos in pos_tags if pos in complex_pos)
    
    # Clause density approximation (complex POS per sentence)
    clause_density = complex_count / max(1, word_count) * 100
    
    # Sentence length normalized score
    length_score = min(word_count / 20.0, 1.0)  # Normalize to 0-1, cap at 20 words
    
    # Combined complexity score
    complexity_score = (length_score * 0.4 + 
                       (avg_word_length / 10.0) * 0.3 + 
                       (clause_density / 10.0) * 0.3)
    
    return {
        'word_count': word_count,
        'avg_word_length': avg_word_length,
        'complex_pos_count': complex_count,
        'clause_density': clause_density,
        'complexity_score': complexity_score
    }

def generate_sentences_by_complexity():
    """Generate sentences with controlled syntactic complexity"""
    
    # Simple sentences (complexity level 1)
    simple_templates = [
        "The {} is {}.",
        "I {} the {}.",
        "She {} {}.",
        "We {} {}.",
        "They {} the {}.",
        "The {} {} quickly.",
        "My {} is {}.",
        "This {} looks {}.",
        "The {} runs fast.",
        "Birds {} high."
    ]
    
    # Medium complexity sentences (complexity level 2)
    medium_templates = [
        "The {} that I {} is very {}.",
        "When the {} arrives, we will {} the {}.",
        "Although the {} is {}, I still {} it.",
        "The {} which was {} became {} yesterday.",
        "If the {} is {}, then we should {} it.",
        "Because the {} was {}, everyone {} it.",
        "The {} who {} the {} is very {}.",
        "While the {} was {}, the {} remained {}.",
        "Since the {} became {}, we have {} it.",
        "After the {} was {}, they {} the {}."
    ]
    
    # Complex sentences (complexity level 3)
    complex_templates = [
        "The {} that was {} by the {} who {} it became {} when the {} decided to {} it.",
        "Although the {} which had been {} was {}, the {} that {} it remained {} because the {} was {}.",
        "When the {} who {} the {} that was {} realized that the {} was {}, they {} it.",
        "If the {} that {} the {} which was {} by the {} becomes {}, then the {} will {}.",
        "Because the {} which had {} the {} that was {} decided to {} it, the {} became {}.",
        "The {} who {} the {} that had been {} by the {} was {} when the {} {}.",
        "While the {} that {} the {} was {}, the {} which had been {} remained {}.",
        "Since the {} who had {} the {} became {}, the {} that was {} started to {}.",
        "After the {} which was {} by the {} that {} it became {}, they {} the {}.",
        "Unless the {} that {} the {} which was {} becomes {}, the {} will remain {}."
    ]
    
    # Word lists for filling templates
    nouns = ['cat', 'dog', 'book', 'car', 'house', 'tree', 'bird', 'phone', 'computer', 'table', 
             'chair', 'window', 'door', 'garden', 'mountain', 'river', 'ocean', 'sky', 'sun', 'moon']
    verbs = ['runs', 'walks', 'sits', 'stands', 'flies', 'swims', 'reads', 'writes', 'sings', 'dances',
             'plays', 'works', 'sleeps', 'eats', 'drinks', 'thinks', 'speaks', 'listens', 'watches', 'helps']
    adjectives = ['big', 'small', 'red', 'blue', 'fast', 'slow', 'happy', 'sad', 'bright', 'dark',
                  'hot', 'cold', 'new', 'old', 'good', 'bad', 'clean', 'dirty', 'loud', 'quiet']
    
    sentences = []
    complexity_levels = []
    
    # Generate simple sentences (50 sentences)
    for _ in range(50):
        template = random.choice(simple_templates)
        try:
            if template.count('{}') == 1:
                sentence = template.format(random.choice(nouns + adjectives))
            elif template.count('{}') == 2:
                sentence = template.format(random.choice(nouns), random.choice(adjectives + verbs))
            else:
                sentence = template.format(random.choice(nouns), random.choice(verbs), random.choice(adjectives))
        except:
            sentence = template.format(*[random.choice(nouns + verbs + adjectives) for _ in range(template.count('{}'))])
        
        sentences.append(sentence)
        complexity_levels.append(1)
    
    # Generate medium complexity sentences (50 sentences)
    for _ in range(50):
        template = random.choice(medium_templates)
        try:
            words_needed = template.count('{}')
            words_list = []
            for i in range(words_needed):
                if i % 3 == 0:
                    words_list.append(random.choice(nouns))
                elif i % 3 == 1:
                    words_list.append(random.choice(verbs))
                else:
                    words_list.append(random.choice(adjectives))
            sentence = template.format(*words_list)
        except:
            sentence = template.format(*[random.choice(nouns + verbs + adjectives) for _ in range(template.count('{}'))])
        
        sentences.append(sentence)
        complexity_levels.append(2)
    
    # Generate complex sentences (50 sentences)
    for _ in range(50):
        template = random.choice(complex_templates)
        try:
            words_needed = template.count('{}')
            words_list = []
            for i in range(words_needed):
                if i % 3 == 0:
                    words_list.append(random.choice(nouns))
                elif i % 3 == 1:
                    words_list.append(random.choice(verbs))
                else:
                    words_list.append(random.choice(adjectives))
            sentence = template.format(*words_list)
        except:
            sentence = template.format(*[random.choice(nouns + verbs + adjectives) for _ in range(template.count('{}'))])
        
        sentences.append(sentence)
        complexity_levels.append(3)
    
    return sentences, complexity_levels

def main():
    print("Starting syntactic complexity embedding analysis...")
    
    # Generate sentences
    print("Generating sentences with controlled syntactic complexity...")
    sentences, complexity_levels = generate_sentences_by_complexity()
    print(f"Generated {len(sentences)} sentences")
    
    # Calculate complexity metrics for verification
    print("Calculating syntactic complexity metrics...")
    complexity_metrics = []
    for sentence in sentences:
        metrics = calculate_syntactic_complexity(sentence)
        complexity_metrics.append(metrics)
    
    # Load embedding model
    print("Loading sentence embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Generate embeddings
    print("Generating sentence embeddings...")
    embeddings = model.encode(sentences)
    print(f"Generated embeddings shape: {embeddings.shape}")
    
    # Clustering analysis
    print("Performing clustering analysis...")
    
    # 1. Cluster by syntactic complexity (true labels)
    n_clusters = 3
    complexity_silhouette_scores = []
    
    # Cluster using true complexity levels
    kmeans_complexity = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    complexity_clusters = kmeans_complexity.fit_predict(embeddings)
    complexity_silhouette = silhouette_score(embeddings, complexity_levels)
    complexity_silhouette_scores.append(complexity_silhouette)
    
    # 2. Random clustering baseline
    random_silhouette_scores = []
    for i in range(5):  # Average over 5 random trials
        random_labels = np.random.randint(0, n_clusters, len(sentences))
        try:
            random_silhouette = silhouette_score(embeddings, random_labels)
            random_silhouette_scores.append(random_silhouette)
        except:
            random_silhouette_scores.append(0.0)
    
    # 3. K-means clustering on embeddings (unsupervised)
    kmeans_embedding = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    embedding_clusters = kmeans_embedding.fit_predict(embeddings)
    embedding_silhouette = silhouette_score(embeddings, embedding_clusters)
    
    # Calculate metrics
    avg_random_silhouette = np.mean(random_silhouette_scores)
    complexity_silhouette_diff = complexity_silhouette - avg_random_silhouette
    embedding_silhouette_diff = embedding_silhouette - avg_random_silhouette
    
    # Analyze complexity metrics by level
    complexity_stats = {}
    for level in [1, 2, 3]:
        level_metrics = [complexity_metrics[i] for i, l in enumerate(complexity_levels) if l == level]
        complexity_stats[f'level_{level}'] = {
            'avg_word_count': np.mean([m['word_count'] for m in level_metrics]),
            'avg_complexity_score': np.mean([m['complexity_score'] for m in level_metrics]),
            'avg_clause_density': np.mean([m['clause_density'] for m in level_metrics]),
            'count': len(level_metrics)
        }
    
    # Results
    results = {
        'experiment': 'syntactic_complexity_embedding_clustering',
        'hypothesis': 'Sentence embeddings cluster differently based on syntactic complexity',
        'n_samples': len(sentences),
        'n_complexity_levels': 3,
        'embedding_model': 'all-MiniLM-L6-v2',
        'clustering_results': {
            'complexity_based_silhouette': float(complexity_silhouette),
            'embedding_based_silhouette': float(embedding_silhouette),
            'random_baseline_silhouette': float(avg_random_silhouette),
            'complexity_vs_random_diff': float(complexity_silhouette_diff),
            'embedding_vs_random_diff': float(embedding_silhouette_diff)
        },
        'complexity_statistics': complexity_stats,
        'success_threshold': 0.15,
        'experiment_success': complexity_silhouette_diff > 0.15,
        'sample_sentences_by_level': {
            'simple': sentences[:5],
            'medium': sentences[50:55],
            'complex': sentences[100:105]
        }
    }
    
    # Print summary
    print("\n" + "="*60)
    print("SYNTACTIC COMPLEXITY EMBEDDING ANALYSIS RESULTS")
    print("="*60)
    print(f"Total sentences analyzed: {len(sentences)}")
    print(f"Embedding model: all-MiniLM-L6-v2")
    print(f"Number of clusters: {n_clusters}")
    print()
    print("Silhouette Scores:")
    print(f"  Complexity-based clustering: {complexity_silhouette:.4f}")
    print(f"  Embedding-based clustering:  {embedding_silhouette:.4f}")
    print(f"  Random baseline (avg):      {avg_random_silhouette:.4f}")
    print()
    print("Differences from random baseline:")
    print(f"  Complexity-based difference: {complexity_silhouette_diff:.4f}")
    print(f"  Embedding-based difference:  {embedding_silhouette_diff:.4f}")
    print()
    print(f"Success threshold: {0.15}")
    print(f"Experiment success: {complexity_silhouette_diff > 0.15}")
    print()
    print("Complexity Statistics by Level:")
    for level, stats in complexity_stats.items():
        print(f"  {level}: {stats['count']} sentences, "
              f"avg_words={stats['avg_word_count']:.1f}, "
              f"complexity_score={stats['avg_complexity_score']:.3f}")
    
    # Save results
    output_path = "/Users/kumacmini/cost-aware-research-search/results/iter_01_results.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    print("Experiment completed successfully!")

if __name__ == "__main__":
    main()