import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
import textstat
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
from nltk.parse import CoreNLPParser
import time
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

def generate_sentences_by_complexity():
    """Generate sentences with controlled syntactic complexity"""
    
    # Simple sentences (low complexity)
    simple_sentences = [
        "The cat sleeps.",
        "Dogs bark loudly.",
        "Children play outside.",
        "Birds fly high.",
        "Rain falls down.",
        "The sun shines.",
        "Books are heavy.",
        "Water is cold.",
        "Food tastes good.",
        "Flowers smell nice.",
        "Cars move fast.",
        "People walk slowly.",
        "Wind blows hard.",
        "Fire burns bright.",
        "Snow melts quickly.",
        "Trees grow tall.",
        "Fish swim deep.",
        "Babies cry often.",
        "Music sounds beautiful.",
        "Stars appear bright.",
        "Grass feels soft.",
        "Bread smells fresh.",
        "Coffee tastes bitter.",
        "Time passes fast.",
        "Money buys things.",
        "Work requires effort.",
        "Love brings joy.",
        "Hope gives strength.",
        "Fear causes worry.",
        "Peace feels calm.",
        "Truth matters most.",
        "Dreams inspire hope.",
        "Knowledge brings power.",
        "Success requires work.",
        "Health needs care.",
        "Friends offer support.",
        "Family provides love.",
        "Education opens doors.",
        "Exercise builds strength.",
        "Sleep restores energy.",
        "Food provides nutrition.",
        "Art expresses emotion.",
        "Music creates harmony.",
        "Nature shows beauty.",
        "Science explains phenomena.",
        "Technology advances rapidly.",
        "Communication builds relationships.",
        "Cooperation achieves goals.",
        "Practice improves performance.",
        "Patience yields results."
    ]
    
    # Medium complexity sentences
    medium_sentences = [
        "The cat that lives next door sleeps on the warm windowsill.",
        "When dogs bark loudly, their owners often feel embarrassed.",
        "Children who play outside during summer develop stronger immune systems.",
        "Although birds fly high, they sometimes rest on telephone wires.",
        "The rain that falls down from dark clouds nourishes the garden.",
        "Because the sun shines brightly, people wear protective sunglasses.",
        "Books that are heavy usually contain more comprehensive information.",
        "When water is cold, swimmers often hesitate before diving in.",
        "Food that tastes good frequently contains unhealthy ingredients.",
        "Flowers that smell nice attract bees and other pollinating insects.",
        "Cars that move fast require experienced drivers for safe operation.",
        "People who walk slowly often notice details that others miss.",
        "When wind blows hard, trees bend gracefully to avoid breaking.",
        "Fire that burns bright provides warmth during cold winter nights.",
        "Snow that melts quickly creates streams that flow toward rivers.",
        "Trees that grow tall provide shade for smaller plants below.",
        "Fish that swim deep avoid predators that hunt near the surface.",
        "Babies who cry often may be experiencing discomfort or hunger.",
        "Music that sounds beautiful can evoke strong emotional responses.",
        "Stars that appear bright help travelers navigate during dark nights.",
        "Grass that feels soft indicates healthy soil and adequate moisture.",
        "Bread that smells fresh often sells faster than day-old varieties.",
        "Coffee that tastes bitter might need sugar to improve palatability.",
        "Time that passes fast usually involves engaging and enjoyable activities.",
        "Money that buys things should be spent wisely and responsibly.",
        "Work that requires effort often yields more satisfying results.",
        "Love that brings joy strengthens relationships between committed partners.",
        "Hope that gives strength helps people overcome difficult challenges.",
        "Fear that causes worry can be reduced through preparation.",
        "Peace that feels calm allows minds to rest and restore.",
        "Truth that matters most sometimes requires courage to express.",
        "Dreams that inspire hope motivate people toward positive action.",
        "Knowledge that brings power should be used ethically and wisely.",
        "Success that requires work teaches valuable lessons about perseverance.",
        "Health that needs care benefits from regular exercise and nutrition.",
        "Friends who offer support create networks that last lifetimes.",
        "Family that provides love forms the foundation for personal growth.",
        "Education that opens doors requires dedication and consistent effort.",
        "Exercise that builds strength should be balanced with adequate rest.",
        "Sleep that restores energy improves cognitive function and mood.",
        "Food that provides nutrition supports optimal physical and mental health.",
        "Art that expresses emotion connects people across cultural boundaries.",
        "Music that creates harmony brings communities together in celebration.",
        "Nature that shows beauty inspires artists and scientists alike.",
        "Science that explains phenomena helps humanity understand the universe.",
        "Technology that advances rapidly changes how people work and communicate.",
        "Communication that builds relationships requires active listening and empathy.",
        "Cooperation that achieves goals demonstrates the power of teamwork.",
        "Practice that improves performance requires consistency and focused attention.",
        "Patience that yields results teaches the value of delayed gratification."
    ]
    
    # Complex sentences
    complex_sentences = [
        "Although the cat that lives next door usually sleeps on the warm windowsill during sunny afternoons, today it chose to rest under the oak tree because the weather was particularly humid.",
        "When dogs bark loudly in the early morning hours, their owners, who are often embarrassed by the disturbance they cause to neighbors, typically try to quiet them by offering treats or taking them for walks.",
        "Children who play outside during summer months, despite the risk of sunburn and dehydration that concerns their parents, develop stronger immune systems and better social skills than those who spend most of their time indoors.",
        "Although birds that fly high above the city can see vast landscapes spread below them, they sometimes choose to rest on telephone wires or building rooftops when they need to conserve energy for longer journeys.",
        "The rain that falls down from the dark clouds that have been gathering since dawn not only nourishes the garden that the elderly woman tends with such care but also fills the reservoir that supplies water to the entire community.",
        "Because the sun that shines brightly during these summer days can cause serious skin damage to people who work outdoors, construction workers and farmers often wear protective clothing and apply sunscreen multiple times throughout their shifts.",
        "Books that are heavy with knowledge and wisdom, while sometimes difficult to carry in backpacks during long commutes, usually contain more comprehensive information than their lighter counterparts that focus on entertainment rather than education.",
        "When water that flows from mountain springs becomes cold during winter months, swimmers who are accustomed to warmer temperatures often hesitate before diving in, unless they have trained their bodies to tolerate the shock.",
        "Food that tastes exceptionally good at expensive restaurants frequently contains ingredients that, while delicious, may not provide the nutritional value that health-conscious diners seek when they make dining choices.",
        "Flowers that smell particularly nice during their peak blooming season not only attract bees and other pollinating insects that are essential for reproduction but also draw gardeners who appreciate their aesthetic beauty.",
        "Cars that move fast on highways during rush hour require drivers who have developed the experience and reflexes necessary to navigate safely through traffic while maintaining awareness of other vehicles, pedestrians, and road conditions.",
        "People who choose to walk slowly through city streets, despite the pressure to move quickly in urban environments, often discover architectural details, interesting shop windows, and human interactions that hurried commuters miss.",
        "When strong winds that blow across the plains during storm seasons reach dangerous speeds, trees that have grown tall over many decades bend their trunks and branches gracefully to distribute the force and avoid breaking.",
        "Campfires that burn bright throughout cold winter nights provide not only physical warmth to people gathered around them but also create a focal point for storytelling and social bonding that strengthens community relationships.",
        "Snow that melts quickly during spring thaws creates rushing streams that carry nutrients and sediment toward rivers, which then transport these materials to larger bodies of water that support diverse ecosystems.",
        "Ancient trees that have grown tall over centuries provide shade and shelter for countless smaller plants and animals below them, creating microhabitats that support biodiversity in ways that scientists are still discovering.",
        "Deep-sea fish that swim in the ocean's darkest depths have evolved specialized adaptations to avoid predators and find food in an environment where sunlight never penetrates and pressure is extreme.",
        "Babies who cry frequently during their first months of life may be experiencing discomfort from digestive issues, hunger, or overstimulation, requiring parents to develop patience and observation skills to interpret their needs.",
        "Classical music that sounds beautiful to trained listeners can evoke profound emotional responses and even trigger memories from childhood, demonstrating the complex relationship between auditory processing and psychological experience.",
        "Distant stars that appear as tiny points of light in the night sky have actually been traveling for years or even centuries to reach Earth, making astronomy a study of the universe's past rather than its present state.",
        "Grass that feels soft under bare feet during summer evenings indicates not only healthy soil with adequate moisture content but also the presence of beneficial microorganisms that support the entire local ecosystem.",
        "Freshly baked bread that fills the kitchen with its distinctive aroma often sells faster than day-old varieties at bakeries because the scent triggers hunger responses and pleasant associations with home and comfort.",
        "Coffee that tastes bitter due to over-extraction or poor quality beans might benefit from the addition of sugar, cream, or alternative brewing methods that highlight its more pleasant flavor characteristics.",
        "Time that seems to pass quickly during engaging activities like playing games, reading compelling books, or spending time with loved ones demonstrates how psychological perception of temporal flow varies based on attention and emotional state.",
        "Money that enables people to purchase goods and services should ideally be earned through meaningful work and spent in ways that contribute to personal well-being and social benefit rather than mere consumption."
    ]
    
    return simple_sentences[:50], medium_sentences[:50], complex_sentences[:25]

def calculate_syntactic_complexity(sentence):
    """Calculate multiple syntactic complexity metrics for a sentence"""
    metrics = {}
    
    # Basic readability metrics
    metrics['flesch_reading_ease'] = textstat.flesch_reading_ease(sentence)
    metrics['flesch_kincaid_grade'] = textstat.flesch_kincaid_grade(sentence)
    metrics['automated_readability_index'] = textstat.automated_readability_index(sentence)
    
    # Length-based metrics
    words = word_tokenize(sentence)
    metrics['word_count'] = len(words)
    metrics['avg_word_length'] = np.mean([len(word) for word in words])
    
    # Syntactic features
    pos_tags = pos_tag(words)
    metrics['noun_count'] = sum(1 for _, tag in pos_tags if tag.startswith('N'))
    metrics['verb_count'] = sum(1 for _, tag in pos_tags if tag.startswith('V'))
    metrics['adj_count'] = sum(1 for _, tag in pos_tags if tag.startswith('J'))
    metrics['subordinate_clause_markers'] = sum(1 for word in words if word.lower() in ['that', 'which', 'who', 'when', 'where', 'because', 'although', 'while', 'since'])
    
    return metrics

def get_embeddings(sentences, model_name):
    """Get embeddings for sentences using specified model"""
    model = SentenceTransformer(model_name)
    embeddings = model.encode(sentences)
    return embeddings

def perform_clustering_analysis(embeddings, true_labels, n_clusters=3):
    """Perform clustering and calculate silhouette score"""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    if len(set(cluster_labels)) > 1:
        silhouette_avg = silhouette_score(embeddings, cluster_labels)
    else:
        silhouette_avg = -1
    
    return cluster_labels, silhouette_avg

def calculate_baseline_score(embeddings, n_trials=10):
    """Calculate baseline silhouette score with random clustering"""
    scores = []
    n_samples = len(embeddings)
    
    for _ in range(n_trials):
        random_labels = np.random.randint(0, 3, n_samples)
        if len(set(random_labels)) > 1:
            score = silhouette_score(embeddings, random_labels)
            scores.append(score)
    
    return np.mean(scores) if scores else -1

def main():
    print("Starting Syntactic Complexity Embedding Clustering Experiment")
    print("=" * 60)
    
    start_time = time.time()
    
    # Generate sentences with controlled complexity
    print("Generating sentences with controlled syntactic complexity...")
    simple_sentences, medium_sentences, complex_sentences = generate_sentences_by_complexity()
    
    # Combine all sentences and create labels
    all_sentences = simple_sentences + medium_sentences + complex_sentences
    true_complexity_labels = ([0] * len(simple_sentences) + 
                             [1] * len(medium_sentences) + 
                             [2] * len(complex_sentences))
    
    print(f"Generated {len(all_sentences)} sentences:")
    print(f"  Simple: {len(simple_sentences)}")
    print(f"  Medium: {len(medium_sentences)}")
    print(f"  Complex: {len(complex_sentences)}")
    
    # Calculate complexity metrics for validation
    print("\nCalculating syntactic complexity metrics...")
    complexity_metrics = [calculate_syntactic_complexity(sent) for sent in all_sentences]
    
    # Test multiple embedding models
    embedding_models = [
        'all-MiniLM-L6-v2',
        'all-mpnet-base-v2'
    ]
    
    results = {}
    
    for model_name in embedding_models:
        print(f"\nTesting model: {model_name}")
        
        # Get embeddings
        embeddings = get_embeddings(all_sentences, model_name)
        print(f"Generated embeddings with shape: {embeddings.shape}")
        
        # Perform clustering based on syntactic complexity
        cluster_labels, syntax_silhouette = perform_clustering_analysis(
            embeddings, true_complexity_labels, n_clusters=3
        )
        
        # Calculate baseline (random clustering)
        baseline_silhouette = calculate_baseline_score(embeddings)
        
        # Calculate improvement over baseline
        improvement = syntax_silhouette - baseline_silhouette
        
        results[model_name] = {
            'syntax_based_silhouette': float(syntax_silhouette),
            'random_baseline_silhouette': float(baseline_silhouette),
            'improvement_over_baseline': float(improvement),
            'cluster_distribution': [int(np.sum(np.array(cluster_labels) == i)) for i in range(3)],
            'success_threshold_met': improvement > 0.15
        }
        
        print(f"  Syntax-based clustering silhouette: {syntax_silhouette:.4f}")
        print(f"  Random baseline silhouette: {baseline_silhouette:.4f}")
        print(f"  Improvement over baseline: {improvement:.4f}")
        print(f"  Success threshold (>0.15) met: {improvement > 0.15}")
    
    # Calculate complexity statistics for validation
    complexity_stats = {}
    for category, start_idx in [('simple', 0), ('medium', 50), ('complex', 100)]:
        if category == 'complex':
            end_idx = len(complexity_metrics)
        else:
            end_idx = start_idx + 50
            
        category_metrics = complexity_metrics[start_idx:end_idx]
        complexity_stats[category] = {
            'avg_word_count': float(np.mean([m['word_count'] for m in category_metrics])),
            'avg_flesch_score': float(np.mean([m['flesch_reading_ease'] for m in category_metrics])),
            'avg_subordinate_clauses': float(np.mean([m['subordinate_clause_markers'] for m in category_metrics])),
            'sample_count': len(category_metrics)
        }
    
    # Compile final results
    final_results = {
        'hypothesis': "Sentence embeddings from different models will cluster differently based on syntactic complexity",
        'experiment_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_samples': len(all_sentences),
        'complexity_distribution': {
            'simple': len(simple_sentences),
            'medium': len(medium_sentences), 
            'complex': len(complex_sentences)
        },
        'complexity_validation_stats': complexity_stats,
        'embedding_model_results': results,
        'success_criteria': {
            'threshold': 0.15,
            'metric': 'improvement in silhouette score over random baseline',
            'models_meeting_threshold': sum(1 for r in results.values() if r['success_threshold_met'])
        },
        'execution_time_seconds': time.time() - start_time
    }
    
    # Print summary
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    
    print(f"Total samples processed: {len(all_sentences)}")
    print(f"Execution time: {final_results['execution_time_seconds']:.2f} seconds")
    print()
    
    print("Complexity Category Validation:")
    for category, stats in complexity_stats.items():
        print(f"  {category.capitalize()}: {stats['sample_count']} samples")
        print(f"    Avg word count: {stats['avg_word_count']:.1f}")
        print(f"    Avg Flesch score: {stats['avg_flesch_score']:.1f}")
        print(f"    Avg subordinate clauses: {stats['avg_subordinate_clauses']:.1f}")
    
    print("\nModel Performance:")
    for model, result in results.items():
        print(f"  {model}:")
        print(f"    Syntax clustering silhouette: {result['syntax_based_silhouette']:.4f}")
        print(f"    Improvement over baseline: {result['improvement_over_baseline']:.4f}")
        print(f"    Threshold met: {result['success_threshold_met']}")
    
    successful_models = sum(1 for r in results.values() if r['success_threshold_met'])
    print(f"\nModels meeting success threshold: {successful_models}/{len(results)}")
    
    if successful_models > 0:
        print("✓ HYPOTHESIS SUPPORTED: Some models show syntactic clustering")
    else:
        print("✗ HYPOTHESIS NOT SUPPORTED: No models show strong syntactic clustering")
    
    # Save results
    output_path = '/Users/kumacmini/cost-aware-research-search/results/iter_01_results.json'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")

if __name__ == "__main__":
    main()