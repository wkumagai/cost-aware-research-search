import json
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import random
import time
from collections import defaultdict
import os
import logging

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def jaccard_similarity(text1, text2):
    """Calculate Jaccard similarity between two texts"""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    return len(intersection) / len(union) if len(union) > 0 else 0

def lexical_overlap_score(text1, text2):
    """Calculate lexical overlap as proportion of shared words"""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    if len(words1) == 0 and len(words2) == 0:
        return 1.0
    return len(words1.intersection(words2)) / max(len(words1), len(words2))

def bootstrap_correlation(x, y, n_bootstrap=1000):
    """Calculate bootstrap confidence intervals for Spearman correlation"""
    correlations = []
    n = len(x)
    
    for _ in range(n_bootstrap):
        indices = np.random.choice(n, n, replace=True)
        x_boot = np.array(x)[indices]
        y_boot = np.array(y)[indices]
        corr, _ = spearmanr(x_boot, y_boot)
        if not np.isnan(corr):
            correlations.append(corr)
    
    correlations = np.array(correlations)
    ci_lower = np.percentile(correlations, 2.5)
    ci_upper = np.percentile(correlations, 97.5)
    return np.mean(correlations), ci_lower, ci_upper

def cohens_d(group1, group2):
    """Calculate Cohen's d effect size"""
    n1, n2 = len(group1), len(group2)
    pooled_std = np.sqrt(((n1 - 1) * np.var(group1, ddof=1) + (n2 - 1) * np.var(group2, ddof=1)) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0
    return (np.mean(group1) - np.mean(group2)) / pooled_std

def generate_sentence_pairs():
    """Generate diverse sentence pairs with graded similarity labels"""
    pairs = []
    
    # High similarity pairs (score 4-5)
    high_sim_pairs = [
        ("The cat is sleeping on the couch", "A cat sleeps on the sofa", 4.8),
        ("I went to the store yesterday", "Yesterday I visited the shop", 4.5),
        ("The weather is beautiful today", "Today has wonderful weather", 4.7),
        ("She loves reading mystery books", "Mystery novels are her favorite reading", 4.6),
        ("The dog ran quickly through the park", "A dog sprinted fast across the park", 4.9),
        ("He plays guitar every evening", "Every night he practices guitar", 4.5),
        ("The movie was incredibly boring", "That film was extremely dull", 4.7),
        ("We need to finish this project soon", "This project requires quick completion", 4.3),
        ("The restaurant serves excellent food", "This place has amazing cuisine", 4.4),
        ("Children are playing in the garden", "Kids play outside in the yard", 4.2),
        ("The train arrives at noon", "At twelve, the train comes", 4.6),
        ("She works as a software engineer", "Her job is in software development", 4.3),
        ("The book contains fascinating stories", "This volume has captivating tales", 4.5),
        ("My car needs urgent repairs", "The vehicle requires immediate fixing", 4.4),
        ("Students study hard for exams", "Pupils work diligently before tests", 4.2),
    ]
    
    # Medium similarity pairs (score 2-3.5)
    medium_sim_pairs = [
        ("The sun is shining brightly", "It's a hot summer day", 3.2),
        ("He bought a new computer", "She purchased some electronics", 2.8),
        ("The meeting starts at 3 PM", "There's a conference this afternoon", 3.0),
        ("I enjoy listening to classical music", "Orchestra concerts are entertaining", 3.4),
        ("The house has five bedrooms", "This building contains multiple rooms", 2.9),
        ("She drives a red sports car", "He owns an expensive vehicle", 2.7),
        ("The company hired new employees", "Several workers joined the organization", 3.3),
        ("Pizza is my favorite food", "Italian cuisine tastes delicious", 3.1),
        ("The library closes at 6 PM", "Books must be returned by evening", 2.6),
        ("He travels frequently for work", "Business requires constant movement", 2.8),
        ("The garden needs more water", "Plants require regular maintenance", 3.0),
        ("She teaches mathematics at university", "Academic instruction involves numbers", 2.9),
        ("The phone battery is dying", "Electronic devices need charging", 2.5),
        ("Winter brings cold temperatures", "Seasonal weather affects everyone", 2.7),
        ("The concert was sold out", "Popular events attract large crowds", 2.4),
    ]
    
    # Low similarity pairs (score 0-2)
    low_sim_pairs = [
        ("The elephant is gray", "Mathematics is difficult", 0.1),
        ("I like ice cream", "The building is tall", 0.0),
        ("Birds can fly", "Computers process data", 0.2),
        ("The ocean is deep", "My birthday is tomorrow", 0.0),
        ("Pizza has cheese", "The moon is bright", 0.1),
        ("Dogs are loyal pets", "Traffic jams cause delays", 0.0),
        ("Books contain knowledge", "Flowers smell beautiful", 0.2),
        ("Rain makes roads wet", "Music helps relaxation", 0.1),
        ("Fire is hot", "Doctors help patients", 0.0),
        ("Trees produce oxygen", "Shopping malls are crowded", 0.1),
        ("Coffee keeps people awake", "Mountains are very high", 0.0),
        ("Fish live in water", "Clocks measure time", 0.2),
        ("Shoes protect feet", "Stars shine at night", 0.1),
        ("Bread is baked food", "Cars need gasoline", 0.0),
        ("Snow is white and cold", "Libraries contain books", 0.1),
    ]
    
    # Paraphrase pairs (high similarity, low lexical overlap)
    paraphrase_pairs = [
        ("The feline rested on the furniture", "A cat napped on the chair", 4.8),
        ("Precipitation fell from the sky", "It was raining outside", 4.9),
        ("The automobile requires fuel", "The car needs gas", 4.7),
        ("Canines are loyal companions", "Dogs make faithful friends", 4.6),
        ("The infant was crying loudly", "A baby wailed noisily", 4.5),
        ("Educational institutions teach students", "Schools educate children", 4.4),
        ("The physician examined the patient", "A doctor checked the sick person", 4.3),
        ("Precipitation began during evening hours", "Rain started after sunset", 4.2),
        ("The residence has multiple levels", "This house is two stories", 4.1),
        ("Culinary experts prepare meals", "Chefs cook delicious food", 4.0),
        ("Transportation vehicles cause pollution", "Cars create environmental problems", 3.8),
        ("Financial institutions store money", "Banks keep people's savings", 3.9),
        ("Communication devices connect people", "Phones help us talk remotely", 3.7),
        ("Recreational activities improve health", "Sports and games boost fitness", 3.6),
        ("Academic assessments measure knowledge", "Tests check what students learned", 3.5),
    ]
    
    # Similar topic, different specifics (medium similarity)
    topic_pairs = [
        ("I visited Paris last summer", "She went to France for vacation", 3.5),
        ("The iPhone has good camera quality", "Smartphones take decent photos", 3.2),
        ("Harvard is a prestigious university", "Elite colleges have high standards", 3.4),
        ("Tesla makes electric vehicles", "Car companies produce automobiles", 3.0),
        ("Amazon sells books online", "E-commerce platforms offer products", 2.9),
        ("Netflix streams movies and shows", "Entertainment platforms provide content", 3.1),
        ("Google processes search queries", "Internet companies handle user requests", 3.3),
        ("McDonald's serves fast food", "Restaurants provide quick meals", 3.2),
        ("Apple develops innovative technology", "Tech firms create new devices", 2.8),
        ("Spotify plays music streaming", "Audio platforms deliver songs", 3.6),
        ("Facebook connects social networks", "Social media links communities", 3.7),
        ("YouTube hosts video content", "Video platforms share multimedia", 3.4),
        ("Microsoft creates software products", "Companies develop computer programs", 2.7),
        ("Nike manufactures athletic shoes", "Sports brands make exercise gear", 3.3),
        ("Starbucks brews coffee drinks", "Cafes serve hot beverages", 3.1),
    ]
    
    # Combine all pairs
    all_pairs = high_sim_pairs + medium_sim_pairs + low_sim_pairs + paraphrase_pairs + topic_pairs
    
    # Shuffle to randomize order
    random.shuffle(all_pairs)
    
    # Take first 120 pairs to meet requirement
    selected_pairs = all_pairs[:120]
    
    for i, (sent1, sent2, score) in enumerate(selected_pairs):
        pairs.append({
            'id': f'pair_{i:03d}',
            'sentence1': sent1,
            'sentence2': sent2,
            'similarity_score': score,
            'lexical_overlap': lexical_overlap_score(sent1, sent2),
            'pair_type': 'paraphrase' if i < len(paraphrase_pairs) else 'semantic'
        })
    
    return pairs

def calculate_embeddings(sentences, model_name):
    """Calculate sentence embeddings using specified model"""
    try:
        model = SentenceTransformer(model_name)
        embeddings = model.encode(sentences)
        return embeddings
    except Exception as e:
        logger.warning(f"Failed to load {model_name}: {e}")
        return None

def calculate_baseline_similarities(pairs):
    """Calculate baseline similarity metrics"""
    sentences1 = [p['sentence1'] for p in pairs]
    sentences2 = [p['sentence2'] for p in pairs]
    
    # TF-IDF cosine similarity
    tfidf = TfidfVectorizer().fit(sentences1 + sentences2)
    tfidf1 = tfidf.transform(sentences1)
    tfidf2 = tfidf.transform(sentences2)
    tfidf_similarities = [cosine_similarity(tfidf1[i:i+1], tfidf2[i:i+1])[0][0] 
                         for i in range(len(pairs))]
    
    # Jaccard similarity
    jaccard_similarities = [jaccard_similarity(s1, s2) 
                           for s1, s2 in zip(sentences1, sentences2)]
    
    # Random baseline
    random_similarities = [random.random() for _ in pairs]
    
    return {
        'tfidf_cosine': tfidf_similarities,
        'jaccard': jaccard_similarities,
        'random': random_similarities
    }

def evaluate_embeddings(embeddings, pairs):
    """Evaluate embedding performance"""
    if embeddings is None:
        return None
    
    # Calculate cosine similarities between embedding pairs
    similarities = []
    for i in range(len(pairs)):
        sim = cosine_similarity([embeddings[i]], [embeddings[i + len(pairs)//2]])[0][0]
        similarities.append(sim)
    
    # Get human similarity scores
    human_scores = [p['similarity_score'] for p in pairs[:len(similarities)]]
    
    # Calculate correlations
    spearman_corr, spearman_p = spearmanr(similarities, human_scores)
    pearson_corr, pearson_p = stats.pearsonr(similarities, human_scores)
    
    # Bootstrap confidence intervals
    boot_corr, ci_lower, ci_upper = bootstrap_correlation(similarities, human_scores)
    
    return {
        'similarities': similarities,
        'spearman_correlation': spearman_corr,
        'spearman_p_value': spearman_p,
        'pearson_correlation': pearson_corr, 
        'pearson_p_value': pearson_p,
        'bootstrap_correlation': boot_corr,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper
    }

def stratified_analysis(results, pairs):
    """Perform stratified analysis by lexical overlap"""
    lexical_overlaps = [p['lexical_overlap'] for p in pairs]
    
    # Define quartiles
    quartiles = np.percentile(lexical_overlaps, [25, 50, 75])
    
    stratified_results = {}
    
    for method, method_results in results.items():
        if method_results is None or 'similarities' not in method_results:
            continue
            
        similarities = method_results['similarities']
        human_scores = [p['similarity_score'] for p in pairs[:len(similarities)]]
        overlaps = lexical_overlaps[:len(similarities)]
        
        # Stratify by quartiles
        for i, (q_name, condition) in enumerate([
            ('low', lambda x: x <= quartiles[0]),
            ('medium_low', lambda x: quartiles[0] < x <= quartiles[1]),
            ('medium_high', lambda x: quartiles[1] < x <= quartiles[2]),
            ('high', lambda x: x > quartiles[2])
        ]):
            indices = [j for j, overlap in enumerate(overlaps) if condition(overlap)]
            
            if len(indices) > 3:  # Need minimum samples
                strat_sims = [similarities[j] for j in indices]
                strat_human = [human_scores[j] for j in indices]
                
                corr, p_val = spearmanr(strat_sims, strat_human)
                
                if method not in stratified_results:
                    stratified_results[method] = {}
                
                stratified_results[method][f'lexical_overlap_{q_name}'] = {
                    'spearman_correlation': corr,
                    'p_value': p_val,
                    'n_samples': len(indices)
                }
    
    return stratified_results

def calculate_effect_sizes(embedding_results, baseline_results):
    """Calculate Cohen's d effect sizes comparing embeddings to baselines"""
    effect_sizes = {}
    
    for emb_model, emb_result in embedding_results.items():
        if emb_result is None:
            continue
            
        effect_sizes[emb_model] = {}
        emb_sims = emb_result['similarities']
        
        for baseline_name, baseline_sims in baseline_results.items():
            if len(baseline_sims) == len(emb_sims):
                effect_size = cohens_d(emb_sims, baseline_sims)
                effect_sizes[emb_model][f'vs_{baseline_name}'] = effect_size
    
    return effect_sizes

def main():
    """Main experiment function"""
    start_time = time.time()
    
    logger.info("Starting comprehensive embedding evaluation experiment")
    
    # Generate sentence pairs
    logger.info("Generating sentence pairs...")
    pairs = generate_sentence_pairs()
    logger.info(f"Generated {len(pairs)} sentence pairs")
    
    # Prepare sentences for embedding
    all_sentences = []
    for pair in pairs:
        all_sentences.extend([pair['sentence1'], pair['sentence2']])
    
    # Calculate baseline similarities
    logger.info("Calculating baseline similarities...")
    baseline_results = calculate_baseline_similarities(pairs)
    
    # Test multiple embedding models
    embedding_models = [
        'all-MiniLM-L6-v2',
        'all-mpnet-base-v2'
    ]
    
    embedding_results = {}
    
    for model_name in embedding_models:
        logger.info(f"Evaluating {model_name}...")
        embeddings = calculate_embeddings(all_sentences, model_name)
        if embeddings is not None:
            # Split embeddings back to pairs
            n_pairs = len(pairs)
            emb1 = embeddings[:n_pairs]
            emb2 = embeddings[n_pairs:n_pairs*2]
            
            # Calculate similarities
            similarities = []
            for i in range(n_pairs):
                sim = cosine_similarity([emb1[i]], [emb2[i]])[0][0]
                similarities.append(sim)
            
            # Get human scores
            human_scores = [p['similarity_score'] for p in pairs]
            
            # Calculate metrics
            spearman_corr, spearman_p = spearmanr(similarities, human_scores)
            pearson_corr, pearson_p = stats.pearsonr(similarities, human_scores)
            boot_corr, ci_lower, ci_upper = bootstrap_correlation(similarities, human_scores)
            
            embedding_results[model_name] = {
                'similarities': similarities,
                'spearman_correlation': spearman_corr,
                'spearman_p_value': spearman_p,
                'pearson_correlation': pearson_corr,
                'pearson_p_value': pearson_p,
                'bootstrap_correlation': boot_corr,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper
            }
    
    # Evaluate baselines
    baseline_evaluations = {}
    human_scores = [p['similarity_score'] for p in pairs]
    
    for baseline_name, baseline_sims in baseline_results.items():
        spearman_corr, spearman_p = spearmanr(baseline_sims, human_scores)
        boot_corr, ci_lower, ci_upper = bootstrap_correlation(baseline_sims, human_scores)
        
        baseline_evaluations[baseline_name] = {
            'similarities': baseline_sims,
            'spearman_correlation': spearman_corr,
            'spearman_p_value': spearman_p,
            'bootstrap_correlation': boot_corr,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper
        }
    
    # Perform stratified analysis
    logger.info("Performing stratified analysis...")
    all_results = {**embedding_results, **baseline_evaluations}
    stratified_results = stratified_analysis(all_results, pairs)
    
    # Calculate effect sizes
    logger.info("Calculating effect sizes...")
    effect_sizes = calculate_effect_sizes(embedding_results, baseline_results)
    
    # Compile final results
    results = {
        'experiment_metadata': {
            'n_pairs': len(pairs),
            'n_models': len(embedding_models),
            'n_baselines': len(baseline_results),
            'runtime_seconds': time.time() - start_time,
            'random_seed': 42,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        },
        'sentence_pairs': pairs,
        'embedding_results': embedding_results,
        'baseline_results': baseline_evaluations,
        'stratified_analysis': stratified_results,
        'effect_sizes': effect_sizes,
        'summary_statistics': {
            'best_embedding_spearman': max([r.get('spearman_correlation', 0) for r in embedding_results.values()]),
            'best_baseline_spearman': max([r.get('spearman_correlation', 0) for r in baseline_evaluations.values()]),
            'embedding_vs_best_baseline_improvement': max([r.get('spearman_correlation', 0) for r in embedding_results.values()]) - max([r.get('spearman_correlation', 0) for r in baseline_evaluations.values()])
        }
    }
    
    # Print summary table
    print("\n" + "="*80)
    print("COMPREHENSIVE EMBEDDING EVALUATION RESULTS")
    print("="*80)
    
    print(f"\nDataset: {len(pairs)} sentence pairs with graded similarity labels (0-5)")
    print(f"Embedding Models: {len(embedding_models)}")
    print(f"Baseline Methods: {len(baseline_results)}")
    
    print("\nMAIN RESULTS (Spearman Correlation with 95% CI):")
    print("-" * 60)
    
    # Sort results by correlation
    all_methods = []
    for method, result in embedding_results.items():
        all_methods.append((method, result, 'embedding'))
    for method, result in baseline_evaluations.items():
        all_methods.append((method, result, 'baseline'))
    
    all_methods.sort(key=lambda x: x[1].get('spearman_correlation', 0), reverse=True)
    
    for method, result, method_type in all_methods:
        corr = result.get('spearman_correlation', 0)
        ci_low = result.get('ci_lower', 0)
        ci_high = result.get('ci_upper', 0)
        p_val = result.get('spearman_p_value', 1)
        
        significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        
        print(f"{method:20s} ({method_type:9s}): r = {corr:.3f} [{ci_low:.3f}, {ci_high:.3f}] {significance}")
    
    print(f"\nHYPOTHESIS TEST:")
    print(f"Target: Spearman r > 0.7 for embedding models")
    
    hypothesis_met = False
    for method, result in embedding_results.items():
        corr = result.get('spearman_correlation', 0)
        if corr > 0.7:
            print(f"✓ {method}: r = {corr:.3f} > 0.7 - HYPOTHESIS SUPPORTED")
            hypothesis_met = True
        else:
            print(f"✗ {method}: r = {corr:.3f} ≤ 0.7 - HYPOTHESIS NOT SUPPORTED")
    
    if not hypothesis_met:
        print("OVERALL HYPOTHESIS: NOT SUPPORTED - No embedding model achieved r > 0.7")
    else:
        print("OVERALL HYPOTHESIS: SUPPORTED - At least one embedding model achieved r > 0.7")
    
    print(f"\nEffect Sizes (Cohen's d vs baselines):")
    print("-" * 40)
    for emb_model, effects in effect_sizes.items():
        print(f"{emb_model}:")
        for comparison, d in effects.items():
            magnitude = "large" if abs(d) > 0.8 else "medium" if abs(d) > 0.5 else "small"
            print(f"  {comparison}: d = {d:.3f} ({magnitude})")
    
    # Save results
    output_file = '/Users/kumacmini/cost-aware-research-search/results/iter_04_results.json'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_file}")
    print(f"Experiment completed in {time.time() - start_time:.1f} seconds")

if __name__ == "__main__":
    main()