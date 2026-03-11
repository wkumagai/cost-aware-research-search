#!/usr/bin/env python3
import json
import os
import sys
import time
import math
import random
import re
import collections
import statistics
from datetime import datetime
import numpy as np
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import openai

def setup_api():
    """Setup OpenAI API client with error handling"""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        sys.exit(1)
    
    openai.api_key = api_key
    return openai.OpenAI(api_key=api_key)

def load_embedding_model():
    """Load sentence transformer model for semantic analysis"""
    try:
        print("Loading embedding model...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        return model
    except Exception as e:
        print(f"Error loading embedding model: {e}")
        sys.exit(1)

def get_standardized_prompts():
    """Generate standardized creative writing prompts for consistent evaluation"""
    base_prompts = [
        "Write a short story about a person who discovers they can see colors that don't exist.",
        "Describe a world where gravity works differently than on Earth.",
        "Tell the story of the last bookstore in a digital world.",
        "Write about a character who can taste emotions.",
        "Describe a day in the life of someone who lives backwards through time.",
        "Write a story about a place where music becomes visible.",
        "Tell about a person who collects forgotten words.",
        "Describe a world where dreams are currency.",
        "Write about someone who can hear the thoughts of inanimate objects.",
        "Tell the story of a library that exists between dimensions."
    ]
    return base_prompts

def generate_text_with_temperature(client, prompt, temperature, max_tokens=150):
    """Generate text using OpenAI API with specified temperature"""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_completion_tokens=max_tokens
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"API error at temperature {temperature}: {e}")
        return None

def calculate_lexical_diversity(text):
    """Calculate type-token ratio (TTR) as lexical diversity metric"""
    if not text:
        return 0.0
    
    # Clean and tokenize text
    words = re.findall(r'\b\w+\b', text.lower())
    if len(words) == 0:
        return 0.0
    
    unique_words = len(set(words))
    total_words = len(words)
    
    # Use moving-average TTR to reduce length bias
    if total_words <= 50:
        return unique_words / total_words
    else:
        # Calculate TTR for overlapping windows of 50 words
        window_size = 50
        ttrs = []
        for i in range(total_words - window_size + 1):
            window = words[i:i + window_size]
            ttr = len(set(window)) / len(window)
            ttrs.append(ttr)
        return float(np.mean(ttrs))

def calculate_repetition_penalty(text):
    """Calculate repetition penalty (lower is more repetitive)"""
    if not text:
        return 0.0
    
    words = re.findall(r'\b\w+\b', text.lower())
    if len(words) < 2:
        return 1.0
    
    # Count n-gram repetitions
    bigrams = [(words[i], words[i+1]) for i in range(len(words)-1)]
    if len(bigrams) == 0:
        return 1.0
    
    unique_bigrams = len(set(bigrams))
    total_bigrams = len(bigrams)
    
    return float(unique_bigrams / total_bigrams)

def calculate_semantic_coherence(embedding_model, text):
    """Calculate semantic coherence using sentence embeddings"""
    if not text or len(text.strip()) < 10:
        return 0.0
    
    try:
        # Split text into sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        
        if len(sentences) < 2:
            return 0.5  # Default coherence for single sentence
        
        # Get embeddings
        embeddings = embedding_model.encode(sentences)
        
        # Calculate pairwise cosine similarities
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i+1, len(embeddings)):
                sim = float(cosine_similarity([embeddings[i]], [embeddings[j]])[0][0])
                similarities.append(sim)
        
        return float(np.mean(similarities)) if similarities else 0.0
        
    except Exception as e:
        print(f"Error calculating semantic coherence: {e}")
        return 0.0

def calculate_lexical_overlap_baseline(text1, text2):
    """Simple baseline using lexical overlap (Jaccard similarity)"""
    if not text1 or not text2:
        return 0.0
    
    words1 = set(re.findall(r'\b\w+\b', text1.lower()))
    words2 = set(re.findall(r'\b\w+\b', text2.lower()))
    
    if len(words1) == 0 and len(words2) == 0:
        return 1.0
    
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    return float(intersection / union) if union > 0 else 0.0

def bootstrap_confidence_interval(data, n_bootstrap=1000, confidence=0.95):
    """Calculate bootstrap confidence interval"""
    if len(data) < 2:
        return (0.0, 0.0)
    
    data = [x for x in data if x is not None and not math.isnan(x)]
    if len(data) < 2:
        return (0.0, 0.0)
    
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_means.append(float(np.mean(sample)))
    
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_means, 100 * alpha / 2)
    upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
    
    return (float(lower), float(upper))

def run_temperature_experiment(client, embedding_model):
    """Run the main temperature experiment"""
    print("Starting temperature analysis experiment...")
    
    # Experiment parameters
    temperatures = [0.1, 0.3, 0.5, 0.7, 0.9, 1.2]
    prompts = get_standardized_prompts()
    n_samples_per_temp = max(10, 60 // len(temperatures))  # Ensure minimum 50 total samples
    
    results = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'temperatures': temperatures,
            'n_samples_per_temp': n_samples_per_temp,
            'total_samples': len(temperatures) * n_samples_per_temp,
            'prompts_used': len(prompts)
        },
        'raw_data': [],
        'analysis': {}
    }
    
    print(f"Generating {len(temperatures) * n_samples_per_temp} total samples...")
    
    api_calls = 0
    max_calls = 30
    
    # Generate data
    for temp in temperatures:
        temp_results = {
            'temperature': float(temp),
            'samples': [],
            'coherence_scores': [],
            'diversity_scores': [],
            'repetition_scores': []
        }
        
        print(f"Testing temperature {temp}...")
        
        for i in range(n_samples_per_temp):
            if api_calls >= max_calls:
                print(f"Reached API call limit ({max_calls})")
                break
                
            # Use different prompts cyclically
            prompt = prompts[i % len(prompts)]
            
            # Generate text
            generated_text = generate_text_with_temperature(client, prompt, temp)
            api_calls += 1
            
            if generated_text:
                # Calculate metrics
                diversity = calculate_lexical_diversity(generated_text)
                repetition = calculate_repetition_penalty(generated_text)
                coherence = calculate_semantic_coherence(embedding_model, generated_text)
                
                sample_data = {
                    'prompt_idx': int(i % len(prompts)),
                    'text': str(generated_text),
                    'lexical_diversity': float(diversity),
                    'repetition_penalty': float(repetition),
                    'semantic_coherence': float(coherence),
                    'text_length': int(len(generated_text))
                }
                
                temp_results['samples'].append(sample_data)
                temp_results['coherence_scores'].append(float(coherence))
                temp_results['diversity_scores'].append(float(diversity))
                temp_results['repetition_scores'].append(float(repetition))
                
            time.sleep(0.1)  # Rate limiting
            
        if api_calls >= max_calls:
            break
            
        results['raw_data'].append(temp_results)
    
    print(f"Completed data generation. Used {api_calls} API calls.")
    return results

def analyze_results(results):
    """Perform statistical analysis on results"""
    print("Performing statistical analysis...")
    
    analysis = {
        'temperature_effects': {},
        'correlations': {},
        'statistical_tests': {},
        'confidence_intervals': {}
    }
    
    # Extract data for analysis
    temperatures = []
    coherence_means = []
    diversity_means = []
    repetition_means = []
    
    all_coherence_by_temp = {}
    all_diversity_by_temp = {}
    
    for temp_data in results['raw_data']:
        temp = temp_data['temperature']
        temperatures.append(temp)
        
        coherence_scores = temp_data['coherence_scores']
        diversity_scores = temp_data['diversity_scores']
        repetition_scores = temp_data['repetition_scores']
        
        # Store for statistical tests
        all_coherence_by_temp[temp] = coherence_scores
        all_diversity_by_temp[temp] = diversity_scores
        
        # Calculate means and confidence intervals
        coherence_mean = float(np.mean(coherence_scores)) if coherence_scores else 0.0
        diversity_mean = float(np.mean(diversity_scores)) if diversity_scores else 0.0
        repetition_mean = float(np.mean(repetition_scores)) if repetition_scores else 0.0
        
        coherence_means.append(coherence_mean)
        diversity_means.append(diversity_mean)
        repetition_means.append(repetition_mean)
        
        # Bootstrap confidence intervals
        coherence_ci = bootstrap_confidence_interval(coherence_scores)
        diversity_ci = bootstrap_confidence_interval(diversity_scores)
        
        analysis['confidence_intervals'][str(temp)] = {
            'coherence_ci': [float(coherence_ci[0]), float(coherence_ci[1])],
            'diversity_ci': [float(diversity_ci[0]), float(diversity_ci[1])]
        }
        
        analysis['temperature_effects'][str(temp)] = {
            'coherence_mean': float(coherence_mean),
            'diversity_mean': float(diversity_mean),
            'repetition_mean': float(repetition_mean),
            'n_samples': int(len(coherence_scores))
        }
    
    # Correlation analysis
    if len(temperatures) >= 3:
        try:
            temp_coherence_corr = float(np.corrcoef(temperatures, coherence_means)[0, 1])
            temp_diversity_corr = float(np.corrcoef(temperatures, diversity_means)[0, 1])
            
            analysis['correlations'] = {
                'temperature_coherence': temp_coherence_corr,
                'temperature_diversity': temp_diversity_corr
            }
        except Exception as e:
            print(f"Error calculating correlations: {e}")
            analysis['correlations'] = {'error': str(e)}
    
    # Statistical significance tests
    try:
        # ANOVA for temperature effects
        coherence_values = [score for temp_data in results['raw_data'] 
                          for score in temp_data['coherence_scores']]
        diversity_values = [score for temp_data in results['raw_data'] 
                          for score in temp_data['diversity_scores']]
        
        temp_labels = [temp_data['temperature'] for temp_data in results['raw_data']
                      for _ in temp_data['coherence_scores']]
        
        if len(set(temp_labels)) > 1 and len(coherence_values) > 5:
            # Group coherence scores by temperature
            coherence_groups = []
            diversity_groups = []
            
            for temp_data in results['raw_data']:
                if len(temp_data['coherence_scores']) > 0:
                    coherence_groups.append(temp_data['coherence_scores'])
                    diversity_groups.append(temp_data['diversity_scores'])
            
            if len(coherence_groups) >= 2:
                f_stat_coh, p_val_coh = stats.f_oneway(*coherence_groups)
                f_stat_div, p_val_div = stats.f_oneway(*diversity_groups)
                
                analysis['statistical_tests'] = {
                    'coherence_anova': {
                        'f_statistic': float(f_stat_coh),
                        'p_value': float(p_val_coh)
                    },
                    'diversity_anova': {
                        'f_statistic': float(f_stat_div),
                        'p_value': float(p_val_div)
                    }
                }
        
    except Exception as e:
        print(f"Error in statistical tests: {e}")
        analysis['statistical_tests'] = {'error': str(e)}
    
    return analysis

def print_summary_table(results):
    """Print formatted summary table"""
    print("\n" + "="*80)
    print("TEMPERATURE ANALYSIS RESULTS")
    print("="*80)
    
    print(f"Total samples: {results['metadata']['total_samples']}")
    print(f"Temperatures tested: {results['metadata']['temperatures']}")
    print()
    
    print(f"{'Temp':<6} {'Coherence':<12} {'Diversity':<12} {'Repetition':<12} {'N':<4}")
    print("-" * 48)
    
    for temp_data in results['raw_data']:
        temp = temp_data['temperature']
        n_samples = len(temp_data['coherence_scores'])
        
        if n_samples > 0:
            coh_mean = np.mean(temp_data['coherence_scores'])
            div_mean = np.mean(temp_data['diversity_scores']) 
            rep_mean = np.mean(temp_data['repetition_scores'])
            
            print(f"{temp:<6.1f} {coh_mean:<12.3f} {div_mean:<12.3f} {rep_mean:<12.3f} {n_samples:<4}")
    
    print("\n" + "="*80)
    
    # Print key findings
    if 'analysis' in results and 'correlations' in results['analysis']:
        corrs = results['analysis']['correlations']
        if 'temperature_coherence' in corrs:
            print(f"Temperature-Coherence Correlation: {corrs['temperature_coherence']:.3f}")
        if 'temperature_diversity' in corrs:
            print(f"Temperature-Diversity Correlation: {corrs['temperature_diversity']:.3f}")
    
    print("="*80)

def main():
    """Main execution function"""
    start_time = time.time()
    
    try:
        # Setup
        client = setup_api()
        embedding_model = load_embedding_model()
        
        # Run experiment
        results = run_temperature_experiment(client, embedding_model)
        
        # Analyze results
        analysis = analyze_results(results)
        results['analysis'] = analysis
        
        # Print summary
        print_summary_table(results)
        
        # Save results
        output_file = "/Users/kumacmini/cost-aware-research-search/results/iter_04_results.json"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to: {output_file}")
        
        # Summary statistics
        total_samples = sum(len(temp_data['samples']) for temp_data in results['raw_data'])
        runtime = time.time() - start_time
        
        print(f"\nExperiment completed successfully!")
        print(f"Total samples generated: {total_samples}")
        print(f"Runtime: {runtime:.1f} seconds")
        
        if total_samples < 50:
            print(f"WARNING: Only {total_samples} samples generated (minimum 50 required)")
        
    except Exception as e:
        print(f"Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()