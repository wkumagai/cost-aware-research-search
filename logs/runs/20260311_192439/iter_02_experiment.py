import os
import json
import numpy as np
import time
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def create_sentence_pairs():
    """Create manually curated sentence pairs with known similarity labels"""
    
    # Similar sentence pairs (labeled as 1)
    similar_pairs = [
        ("The cat is sleeping on the couch", "A feline is resting on the sofa"),
        ("I love eating pizza", "Pizza is my favorite food"),
        ("The weather is beautiful today", "Today has gorgeous weather"),
        ("She drives to work every morning", "Every morning she commutes by car"),
        ("The book was very interesting", "That book was fascinating to read"),
        ("He plays guitar in a band", "He's a guitarist in a musical group"),
        ("The movie was entertaining", "That film was quite enjoyable"),
        ("Dogs are loyal companions", "Canines make faithful friends"),
        ("She enjoys reading novels", "Reading fiction books is her hobby"),
        ("The coffee tastes great", "This coffee has excellent flavor"),
        ("Children love playing games", "Kids enjoy recreational activities"),
        ("The car needs gas", "The vehicle requires fuel"),
        ("He works in an office", "His job is at a corporate building"),
        ("The sunset looks amazing", "What a beautiful evening sky"),
        ("She cooks dinner daily", "Every day she prepares the evening meal"),
        ("The ocean is vast", "The sea is enormous"),
        ("Birds fly in the sky", "Avian creatures soar through the air"),
        ("Music helps me relax", "Listening to songs calms me down"),
        ("The garden has flowers", "Blossoms grow in the yard"),
        ("He teaches mathematics", "Math is his teaching subject"),
        ("The house is painted blue", "Blue paint covers the home"),
        ("She wears a red dress", "Her outfit is a crimson gown"),
        ("The phone is ringing", "A call is coming in"),
        ("They walk in the park", "Strolling through the green space"),
        ("The mountain is tall", "That peak reaches great heights"),
        ("Ice cream is cold", "Frozen dessert has low temperature"),
        ("The library has books", "Books fill the reading facility"),
        ("She studies every night", "Nightly studying is her routine"),
        ("The train arrives soon", "The locomotive will be here shortly"),
        ("He builds furniture", "Crafting wooden items is his trade"),
        ("The river flows quickly", "Water moves rapidly in the stream")
    ]
    
    # Dissimilar sentence pairs (labeled as 0)
    dissimilar_pairs = [
        ("The cat is sleeping", "Economic policies affect inflation rates"),
        ("I love pizza", "The mountain peak is covered in snow"),
        ("Beautiful weather today", "Database optimization requires indexing"),
        ("She drives to work", "Quantum mechanics involves wave functions"),
        ("The book was interesting", "Soccer players train every morning"),
        ("He plays guitar", "Chemical reactions produce energy"),
        ("The movie was fun", "Binary trees store data efficiently"),
        ("Dogs are loyal", "Photosynthesis converts sunlight to energy"),
        ("She reads novels", "Volcanic eruptions create new land"),
        ("Coffee tastes great", "Parliamentary procedures govern debates"),
        ("Children play games", "Cellular mitosis divides chromosomes"),
        ("Car needs gas", "Archaeological discoveries reveal history"),
        ("Works in office", "Magnetic fields influence compasses"),
        ("Amazing sunset", "Algebraic equations solve for variables"),
        ("Cooks dinner daily", "Atmospheric pressure affects weather"),
        ("Ocean is vast", "Computer algorithms process information"),
        ("Birds fly high", "Legal contracts require signatures"),
        ("Music relaxes me", "Geological formations indicate age"),
        ("Garden has flowers", "Statistical analysis reveals patterns"),
        ("Teaches mathematics", "Culinary arts combine flavors creatively"),
        ("House painted blue", "Mechanical engineering designs systems"),
        ("Wears red dress", "Astronomical observations map galaxies"),
        ("Phone is ringing", "Biological processes sustain life"),
        ("Walk in park", "Financial markets fluctuate daily"),
        ("Mountain is tall", "Psychological theories explain behavior"),
        ("Ice cream cold", "Architectural designs maximize space"),
        ("Library has books", "Electrical circuits conduct current"),
        ("Studies every night", "Pharmaceutical research develops medicines"),
        ("Train arrives soon", "Agricultural methods improve yields"),
        ("Builds furniture", "Meteorological data predicts storms"),
        ("River flows fast", "Linguistic patterns vary across cultures")
    ]
    
    return similar_pairs, dissimilar_pairs

def simple_sentence_embedding(sentence):
    """Create a simple embedding using word statistics and basic features"""
    words = sentence.lower().split()
    
    # Basic features
    length = len(sentence)
    word_count = len(words)
    avg_word_length = np.mean([len(word) for word in words]) if words else 0
    
    # Character frequency features (a-z)
    char_freq = np.zeros(26)
    for char in sentence.lower():
        if 'a' <= char <= 'z':
            char_freq[ord(char) - ord('a')] += 1
    char_freq = char_freq / max(1, len(sentence))  # normalize
    
    # Common word presence (simple bag of words for common words)
    common_words = ['the', 'is', 'in', 'to', 'and', 'a', 'of', 'it', 'you', 'that', 'he', 'was', 'for', 'on', 'are', 'as', 'with', 'his', 'they', 'i']
    word_features = np.array([1 if word in words else 0 for word in common_words])
    
    # Combine all features
    embedding = np.concatenate([
        [length / 100.0, word_count / 20.0, avg_word_length / 10.0],  # normalized basic stats
        char_freq,  # character frequencies
        word_features  # common word presence
    ])
    
    return embedding

def compute_cosine_distance(emb1, emb2):
    """Compute cosine distance between two embeddings"""
    similarity = cosine_similarity([emb1], [emb2])[0, 0]
    distance = 1 - similarity
    return distance

def run_experiment():
    """Run the complete embedding similarity experiment"""
    print("Starting Sentence Embedding Similarity Experiment")
    print("=" * 50)
    
    start_time = time.time()
    
    try:
        # Create sentence pairs
        print("Creating sentence pairs...")
        similar_pairs, dissimilar_pairs = create_sentence_pairs()
        
        print(f"Created {len(similar_pairs)} similar pairs and {len(dissimilar_pairs)} dissimilar pairs")
        
        # Generate embeddings and compute distances
        similar_distances = []
        dissimilar_distances = []
        
        print("Computing embeddings and distances for similar pairs...")
        for i, (sent1, sent2) in enumerate(similar_pairs):
            try:
                emb1 = simple_sentence_embedding(sent1)
                emb2 = simple_sentence_embedding(sent2)
                distance = compute_cosine_distance(emb1, emb2)
                similar_distances.append(distance)
                
                if i % 10 == 0:
                    print(f"  Processed {i+1}/{len(similar_pairs)} similar pairs")
            except Exception as e:
                print(f"Error processing similar pair {i}: {e}")
                continue
        
        print("Computing embeddings and distances for dissimilar pairs...")
        for i, (sent1, sent2) in enumerate(dissimilar_pairs):
            try:
                emb1 = simple_sentence_embedding(sent1)
                emb2 = simple_sentence_embedding(sent2)
                distance = compute_cosine_distance(emb1, emb2)
                dissimilar_distances.append(distance)
                
                if i % 10 == 0:
                    print(f"  Processed {i+1}/{len(dissimilar_pairs)} dissimilar pairs")
            except Exception as e:
                print(f"Error processing dissimilar pair {i}: {e}")
                continue
        
        # Convert to numpy arrays
        similar_distances = np.array(similar_distances)
        dissimilar_distances = np.array(dissimilar_distances)
        
        # Calculate statistics
        similar_mean = np.mean(similar_distances)
        similar_std = np.std(similar_distances)
        dissimilar_mean = np.mean(dissimilar_distances)
        dissimilar_std = np.std(dissimilar_distances)
        
        # Calculate percentage difference
        if dissimilar_mean > 0:
            percent_difference = ((dissimilar_mean - similar_mean) / dissimilar_mean) * 100
        else:
            percent_difference = 0
        
        success_threshold = 25.0  # 25% lower average distance for similar pairs
        experiment_success = percent_difference >= success_threshold
        
        print("\nResults Summary:")
        print("=" * 30)
        print(f"Similar pairs - Mean distance: {similar_mean:.4f} (±{similar_std:.4f})")
        print(f"Dissimilar pairs - Mean distance: {dissimilar_mean:.4f} (±{dissimilar_std:.4f})")
        print(f"Percentage difference: {percent_difference:.2f}%")
        print(f"Success threshold: {success_threshold}%")
        print(f"Experiment success: {experiment_success}")
        
        # Create visualization
        plt.figure(figsize=(10, 6))
        plt.hist(similar_distances, bins=20, alpha=0.7, label='Similar pairs', color='blue')
        plt.hist(dissimilar_distances, bins=20, alpha=0.7, label='Dissimilar pairs', color='red')
        plt.xlabel('Cosine Distance')
        plt.ylabel('Frequency')
        plt.title('Distribution of Cosine Distances for Similar vs Dissimilar Sentence Pairs')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('/Users/kumacmini/cost-aware-research-search/results/embedding_distance_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Prepare results
        results = {
            'experiment_info': {
                'hypothesis': 'Sentence embeddings will show measurable distance differences between semantically similar vs. dissimilar sentence pairs',
                'timestamp': datetime.now().isoformat(),
                'runtime_seconds': time.time() - start_time,
                'total_samples': len(similar_distances) + len(dissimilar_distances),
                'similar_pairs_count': len(similar_distances),
                'dissimilar_pairs_count': len(dissimilar_distances)
            },
            'metrics': {
                'similar_mean_distance': float(similar_mean),
                'similar_std_distance': float(similar_std),
                'dissimilar_mean_distance': float(dissimilar_mean),
                'dissimilar_std_distance': float(dissimilar_std),
                'percent_difference': float(percent_difference),
                'success_threshold': success_threshold,
                'experiment_success': experiment_success
            },
            'raw_data': {
                'similar_distances': similar_distances.tolist(),
                'dissimilar_distances': dissimilar_distances.tolist()
            },
            'summary_table': {
                'condition': ['Similar pairs', 'Dissimilar pairs'],
                'mean_distance': [float(similar_mean), float(dissimilar_mean)],
                'std_distance': [float(similar_std), float(dissimilar_std)],
                'sample_count': [len(similar_distances), len(dissimilar_distances)]
            }
        }
        
        # Save results
        output_path = '/Users/kumacmini/cost-aware-research-search/results/iter_02_results.json'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")
        print(f"Visualization saved to: /Users/kumacmini/cost-aware-research-search/results/embedding_distance_distribution.png")
        
        return results
        
    except Exception as e:
        print(f"Experiment failed with error: {e}")
        error_results = {
            'experiment_info': {
                'hypothesis': 'Sentence embeddings will show measurable distance differences between semantically similar vs. dissimilar sentence pairs',
                'timestamp': datetime.now().isoformat(),
                'runtime_seconds': time.time() - start_time,
                'error': str(e)
            },
            'metrics': None,
            'experiment_success': False
        }
        
        output_path = '/Users/kumacmini/cost-aware-research-search/results/iter_02_results.json'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(error_results, f, indent=2)
        
        return error_results

if __name__ == "__main__":
    results = run_experiment()
    
    print("\n" + "="*50)
    print("EXPERIMENT COMPLETE")
    print("="*50)
    
    if results['metrics']:
        print(f"Total samples processed: {results['experiment_info']['total_samples']}")
        print(f"Runtime: {results['experiment_info']['runtime_seconds']:.2f} seconds")
        print(f"Success: {results['metrics']['experiment_success']}")
        print(f"Distance reduction: {results['metrics']['percent_difference']:.2f}%")
    else:
        print("Experiment failed - check error details in results file")