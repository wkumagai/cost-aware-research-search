import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import nltk
import textstat
import random
import warnings
warnings.filterwarnings('ignore')

def download_nltk_dependencies():
    """Download required NLTK data"""
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        nltk.download('punkt', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)

def generate_simple_sentences(n=50):
    """Generate simple sentences with low syntactic complexity"""
    simple_templates = [
        "The cat sleeps.",
        "Dogs bark loudly.",
        "Birds fly high.",
        "Rain falls gently.",
        "Children play outside.",
        "Cars move fast.",
        "Trees grow tall.",
        "Fish swim deep.",
        "Wind blows hard.",
        "Stars shine bright.",
        "Books contain knowledge.",
        "Music sounds beautiful.",
        "Flowers smell sweet.",
        "Coffee tastes bitter.",
        "Snow feels cold.",
        "Fire burns hot.",
        "Water flows quickly.",
        "Clouds drift slowly.",
        "Bells ring clearly.",
        "Grass grows green."
    ]
    
    subjects = ["The cat", "The dog", "The bird", "The child", "The car", "The tree", "The book", "The flower"]
    verbs = ["runs", "jumps", "sleeps", "eats", "plays", "moves", "grows", "shines"]
    adjectives = ["quickly", "slowly", "quietly", "loudly", "gently", "softly", "brightly", "dimly"]
    
    sentences = simple_templates[:min(n//2, len(simple_templates))]
    
    # Generate additional simple sentences
    while len(sentences) < n:
        subject = random.choice(subjects)
        verb = random.choice(verbs)
        adverb = random.choice(adjectives)
        sentence = f"{subject} {verb} {adverb}."
        sentences.append(sentence)
    
    return sentences[:n]

def generate_medium_sentences(n=50):
    """Generate medium complexity sentences"""
    templates = [
        "The cat that lives next door sleeps peacefully on the warm windowsill.",
        "When the rain falls, the children play inside their cozy house.",
        "The dog barks loudly because it sees a stranger approaching the gate.",
        "Although it was late, the student continued studying for the important exam.",
        "The bird that built its nest in our garden sings every morning.",
        "Because the weather was perfect, we decided to have a picnic in the park.",
        "The book that I borrowed from the library contains fascinating stories about adventure.",
        "While the music played softly, the couple danced gracefully across the floor.",
        "The car that my father bought last year runs very efficiently on highway roads.",
        "Since the flowers bloomed early, the garden looks especially beautiful this spring."
    ]
    
    conjunctions = ["because", "although", "while", "since", "when", "if", "unless"]
    relative_pronouns = ["that", "which", "who"]
    
    subjects = ["the student", "the teacher", "the musician", "the artist", "the writer"]
    main_clauses = ["worked diligently", "practiced daily", "studied carefully", "performed excellently"]
    sub_clauses = ["the project was challenging", "time was limited", "resources were scarce"]
    
    sentences = templates[:min(n//2, len(templates))]
    
    # Generate additional medium sentences
    while len(sentences) < n:
        if random.choice([True, False]):
            conjunction = random.choice(conjunctions)
            subject = random.choice(subjects)
            main = random.choice(main_clauses)
            sub = random.choice(sub_clauses)
            sentence = f"{conjunction.capitalize()} {sub}, {subject} {main}."
        else:
            subject = random.choice(subjects)
            pronoun = random.choice(relative_pronouns)
            main = random.choice(main_clauses)
            sentence = f"The person {pronoun} {main} achieved great success."
        sentences.append(sentence)
    
    return sentences[:n]

def generate_complex_sentences(n=50):
    """Generate complex sentences with high syntactic complexity"""
    complex_templates = [
        "The researcher, who had been working on the project for several years, finally discovered that the hypothesis, which many had considered unlikely, was actually supported by the data that had been collected through multiple rigorous experiments.",
        "Although the committee members, each of whom brought different perspectives to the discussion, initially disagreed about the proposal, they eventually reached a consensus after considering all the arguments that had been presented during the lengthy deliberation process.",
        "The company's decision to expand internationally, which was announced after months of careful planning and market analysis, surprised many investors who had not anticipated such an aggressive growth strategy during these economically uncertain times.",
        "When the artist, whose previous works had received critical acclaim, unveiled her latest sculpture, the critics were divided in their opinions, with some praising its innovative approach while others questioned whether it represented a departure from her established style.",
        "The novel, which the author had been revising for over a decade, explores themes of identity and belonging through the interconnected stories of characters whose lives, though seemingly unrelated, gradually reveal profound connections that span multiple generations."
    ]
    
    # Generate more complex sentences with multiple clauses
    complex_starters = [
        "Despite the fact that",
        "Although it is widely believed that",
        "While researchers have long assumed that",
        "Given that recent studies have shown that",
        "Notwithstanding the evidence that"
    ]
    
    middle_clauses = [
        "the committee, which consisted of experts from various fields,",
        "the students, who had been preparing for months,",
        "the data, which had been collected over several years,",
        "the results, which surprised even the most experienced researchers,"
    ]
    
    endings = [
        "demonstrated conclusively that previous assumptions were incorrect.",
        "revealed patterns that had never been observed before in similar studies.",
        "provided compelling evidence for theories that had long been controversial.",
        "suggested new directions for future research in this important field."
    ]
    
    sentences = complex_templates[:min(n//2, len(complex_templates))]
    
    while len(sentences) < n:
        starter = random.choice(complex_starters)
        middle = random.choice(middle_clauses)
        ending = random.choice(endings)
        sentence = f"{starter} {middle} {ending}"
        sentences.append(sentence)
    
    return sentences[:n]

def calculate_complexity_metrics(sentence):
    """Calculate various syntactic complexity metrics for a sentence"""
    # Basic readability scores
    flesch_score = textstat.flesch_reading_ease(sentence)
    flesch_kincaid = textstat.flesch_kincaid_grade(sentence)
    
    # Sentence length metrics
    word_count = len(sentence.split())
    char_count = len(sentence)
    
    # NLTK-based metrics
    tokens = nltk.word_tokenize(sentence)
    pos_tags = nltk.pos_tag(tokens)
    
    # Count different POS types
    pos_counts = {}
    for word, pos in pos_tags:
        pos_counts[pos] = pos_counts.get(pos, 0) + 1
    
    # Count subordinating conjunctions and relative pronouns (complexity indicators)
    complex_markers = ['that', 'which', 'who', 'whom', 'whose', 'when', 'where', 'why', 
                      'because', 'although', 'while', 'since', 'if', 'unless', 'whereas']
    complexity_markers = sum(1 for token in tokens if token.lower() in complex_markers)
    
    # Count punctuation (commas, semicolons indicate complexity)
    punctuation_count = sentence.count(',') + sentence.count(';') + sentence.count(':')
    
    return {
        'flesch_score': flesch_score,
        'flesch_kincaid': flesch_kincaid,
        'word_count': word_count,
        'char_count': char_count,
        'pos_diversity': len(pos_counts),
        'complexity_markers': complexity_markers,
        'punctuation_count': punctuation_count,
        'avg_word_length': sum(len(word) for word in tokens) / len(tokens) if tokens else 0
    }

def main():
    start_time = time.time()
    results = {
        'experiment_info': {
            'hypothesis': 'Sentence embeddings will exhibit distinct clustering patterns based on syntactic complexity',
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_samples': 150
        },
        'data_generation': {},
        'complexity_analysis': {},
        'embedding_analysis': {},
        'clustering_results': {},
        'evaluation_metrics': {},
        'visualization_paths': []
    }
    
    try:
        print("Starting Sentence Embedding Complexity Clustering Experiment")
        print("=" * 60)
        
        # Download NLTK dependencies
        download_nltk_dependencies()
        
        # Generate sentences with different complexity levels
        print("Generating sentences...")
        simple_sentences = generate_simple_sentences(50)
        medium_sentences = generate_medium_sentences(50)
        complex_sentences = generate_complex_sentences(50)
        
        all_sentences = simple_sentences + medium_sentences + complex_sentences
        complexity_labels = ['simple'] * 50 + ['medium'] * 50 + ['complex'] * 50
        
        results['data_generation'] = {
            'simple_count': len(simple_sentences),
            'medium_count': len(medium_sentences),
            'complex_count': len(complex_sentences),
            'total_count': len(all_sentences),
            'sample_simple': simple_sentences[:3],
            'sample_medium': medium_sentences[:3],
            'sample_complex': complex_sentences[:3]
        }
        
        print(f"Generated {len(all_sentences)} sentences")
        
        # Calculate complexity metrics
        print("Calculating complexity metrics...")
        complexity_scores = []
        for sentence in all_sentences:
            metrics = calculate_complexity_metrics(sentence)
            complexity_scores.append(metrics)
        
        # Analyze complexity distributions
        simple_metrics = complexity_scores[:50]
        medium_metrics = complexity_scores[50:100]
        complex_metrics = complexity_scores[100:150]
        
        def get_avg_metric(metrics_list, metric_name):
            return np.mean([m[metric_name] for m in metrics_list])
        
        results['complexity_analysis'] = {
            'simple_avg_flesch': get_avg_metric(simple_metrics, 'flesch_score'),
            'medium_avg_flesch': get_avg_metric(medium_metrics, 'flesch_score'),
            'complex_avg_flesch': get_avg_metric(complex_metrics, 'flesch_score'),
            'simple_avg_word_count': get_avg_metric(simple_metrics, 'word_count'),
            'medium_avg_word_count': get_avg_metric(medium_metrics, 'word_count'),
            'complex_avg_word_count': get_avg_metric(complex_metrics, 'word_count'),
            'simple_avg_complexity_markers': get_avg_metric(simple_metrics, 'complexity_markers'),
            'medium_avg_complexity_markers': get_avg_metric(medium_metrics, 'complexity_markers'),
            'complex_avg_complexity_markers': get_avg_metric(complex_metrics, 'complexity_markers')
        }
        
        # Load sentence transformer model
        print("Loading sentence transformer model...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Generate embeddings
        print("Generating sentence embeddings...")
        embeddings = model.encode(all_sentences)
        
        results['embedding_analysis'] = {
            'embedding_dimension': embeddings.shape[1],
            'embedding_shape': list(embeddings.shape)
        }
        
        # Perform clustering analysis
        print("Performing clustering analysis...")
        
        # Cluster each complexity group separately and calculate silhouette scores
        silhouette_scores = {}
        cluster_results = {}
        
        for complexity_type, start_idx, end_idx in [('simple', 0, 50), ('medium', 50, 100), ('complex', 100, 150)]:
            group_embeddings = embeddings[start_idx:end_idx]
            
            # Use K-means with k=5 for each group
            kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(group_embeddings)
            
            # Calculate silhouette score
            if len(np.unique(cluster_labels)) > 1:
                sil_score = silhouette_score(group_embeddings, cluster_labels)
            else:
                sil_score = 0.0
            
            silhouette_scores[complexity_type] = sil_score
            cluster_results[complexity_type] = {
                'cluster_centers': kmeans.cluster_centers_.tolist(),
                'cluster_labels': cluster_labels.tolist(),
                'inertia': kmeans.inertia_,
                'silhouette_score': sil_score
            }
        
        # Calculate baseline (random grouping)
        random_labels = np.random.randint(0, 5, size=50)
        baseline_score = silhouette_score(embeddings[:50], random_labels)
        
        results['clustering_results'] = cluster_results
        results['clustering_results']['baseline_silhouette'] = baseline_score
        
        # Evaluate hypothesis
        simple_score = silhouette_scores['simple']
        complex_score = silhouette_scores['complex']
        
        # Test if complex sentences have 15% lower silhouette score
        threshold_score = simple_score * 0.85
        hypothesis_supported = complex_score < threshold_score
        
        score_difference_pct = ((simple_score - complex_score) / simple_score) * 100 if simple_score > 0 else 0
        
        results['evaluation_metrics'] = {
            'simple_silhouette_score': simple_score,
            'medium_silhouette_score': silhouette_scores['medium'],
            'complex_silhouette_score': complex_score,
            'baseline_silhouette_score': baseline_score,
            'score_difference_percentage': score_difference_pct,
            'threshold_15_percent_lower': threshold_score,
            'hypothesis_supported': hypothesis_supported,
            'success_threshold_met': score_difference_pct >= 15.0
        }
        
        # Create visualization
        print("Creating visualizations...")
        
        # PCA for visualization
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings)
        
        plt.figure(figsize=(12, 8))
        
        # Plot points by complexity
        colors = ['blue', 'green', 'red']
        labels = ['Simple', 'Medium', 'Complex']
        
        for i, (color, label) in enumerate(zip(colors, labels)):
            start_idx = i * 50
            end_idx = (i + 1) * 50
            plt.scatter(embeddings_2d[start_idx:end_idx, 0], 
                       embeddings_2d[start_idx:end_idx, 1], 
                       c=color, label=label, alpha=0.6, s=50)
        
        plt.xlabel(f'PCA Component 1 (explained variance: {pca.explained_variance_ratio_[0]:.3f})')
        plt.ylabel(f'PCA Component 2 (explained variance: {pca.explained_variance_ratio_[1]:.3f})')
        plt.title('Sentence Embeddings by Syntactic Complexity')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        viz_path = '/Users/kumacmini/cost-aware-research-search/results/embedding_clusters.png'
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        results['visualization_paths'].append(viz_path)
        
        # Summary statistics
        print("\n" + "="*60)
        print("EXPERIMENT RESULTS SUMMARY")
        print("="*60)
        print(f"Total sentences analyzed: {len(all_sentences)}")
        print(f"Simple sentences: {len(simple_sentences)}")
        print(f"Medium sentences: {len(medium_sentences)}")
        print(f"Complex sentences: {len(complex_sentences)}")
        print(f"\nComplexity Metrics (Average):")
        print(f"Simple - Flesch Score: {results['complexity_analysis']['simple_avg_flesch']:.1f}")
        print(f"Medium - Flesch Score: {results['complexity_analysis']['medium_avg_flesch']:.1f}")
        print(f"Complex - Flesch Score: {results['complexity_analysis']['complex_avg_flesch']:.1f}")
        print(f"\nClustering Results:")
        print(f"Simple sentences silhouette score: {simple_score:.4f}")
        print(f"Medium sentences silhouette score: {silhouette_scores['medium']:.4f}")
        print(f"Complex sentences silhouette score: {complex_score:.4f}")
        print(f"Baseline (random) silhouette score: {baseline_score:.4f}")
        print(f"\nHypothesis Testing:")
        print(f"Score difference: {score_difference_pct:.1f}%")
        print(f"Threshold (15% lower): {threshold_score:.4f}")
        print(f"Hypothesis supported: {hypothesis_supported}")
        print(f"Success threshold met (≥15% difference): {score_difference_pct >= 15.0}")
        
        # Save results
        results_path = '/Users/kumacmini/cost-aware-research-search/results/iter_01_results.json'
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        
        results['execution_time_seconds'] = time.time() - start_time
        results['status'] = 'completed'
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {results_path}")
        print(f"Visualization saved to: {viz_path}")
        print(f"Total execution time: {results['execution_time_seconds']:.2f} seconds")
        
        return results
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        results['status'] = 'error'
        results['error'] = str(e)
        results['execution_time_seconds'] = time.time() - start_time
        
        # Save error results
        results_path = '/Users/kumacmini/cost-aware-research-search/results/iter_01_results.json'
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results

if __name__ == "__main__":
    main()