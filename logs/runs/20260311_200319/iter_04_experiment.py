import json
import os
import sys
import time
import random
import statistics
import collections
import re
import numpy as np
from openai import OpenAI

def setup_openai_client():
    """Setup OpenAI client with API key from environment"""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    return OpenAI(api_key=api_key)

def get_standardized_prompts():
    """Return standardized creative writing prompts for consistent evaluation"""
    return [
        "Write a short story about a character who discovers an old diary in their attic.",
        "Describe a conversation between two people meeting for the first time at a coffee shop.",
        "Create a dialogue between a parent and child about moving to a new city.",
        "Write about someone's first day at a new job and their interactions with colleagues.",
        "Describe a scene where friends are planning a surprise party.",
        "Write a story about a person who finds an unexpected item in their mailbox.",
        "Create a dialogue between neighbors discussing a community garden project.",
        "Write about someone teaching a skill to another person.",
        "Describe a scene at a local farmers market on a busy Saturday morning.",
        "Write about two characters collaborating on a creative project."
    ]

def calculate_lexical_diversity(text):
    """Calculate type-token ratio as a measure of lexical diversity"""
    if not text or len(text.strip()) == 0:
        return 0.0
    
    # Clean and tokenize text
    words = re.findall(r'\b\w+\b', text.lower())
    if len(words) == 0:
        return 0.0
    
    unique_words = len(set(words))
    total_words = len(words)
    
    # Return type-token ratio
    return unique_words / total_words

def calculate_semantic_coherence(text):
    """Calculate semantic coherence based on sentence structure and repetition patterns"""
    if not text or len(text.strip()) == 0:
        return 0.0
    
    # Split into sentences
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if len(sentences) == 0:
        return 0.0
    
    coherence_score = 0.0
    
    # Check for complete sentences (basic structure)
    complete_sentences = 0
    for sentence in sentences:
        words = re.findall(r'\b\w+\b', sentence)
        if len(words) >= 3:  # Minimum viable sentence length
            complete_sentences += 1
    
    structure_score = complete_sentences / len(sentences) if sentences else 0
    
    # Check for repetitive patterns (lower repetition = higher coherence)
    words = re.findall(r'\b\w+\b', text.lower())
    if len(words) > 0:
        word_counts = collections.Counter(words)
        max_repetition = max(word_counts.values()) if word_counts else 1
        repetition_penalty = min(max_repetition / len(words), 0.5)  # Cap penalty
        repetition_score = 1.0 - repetition_penalty
    else:
        repetition_score = 0.0
    
    # Check for logical flow (sentence length variation)
    if len(sentences) > 1:
        sentence_lengths = [len(re.findall(r'\b\w+\b', s)) for s in sentences]
        length_variance = np.var(sentence_lengths) if len(sentence_lengths) > 1 else 0
        flow_score = min(length_variance / 10.0, 1.0)  # Normalize variance
    else:
        flow_score = 0.5
    
    # Combine scores
    coherence_score = (structure_score * 0.5 + repetition_score * 0.3 + flow_score * 0.2)
    return min(coherence_score, 1.0)

def generate_text_with_temperature(client, prompt, temperature, max_tokens=150):
    """Generate text using OpenAI API with specified temperature"""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_completion_tokens=max_tokens
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"API error at temperature {temperature}: {e}")
        return ""

def bootstrap_confidence_interval(data, statistic_func, n_bootstrap=1000, confidence=0.95):
    """Calculate bootstrap confidence interval for a statistic"""
    if len(data) == 0:
        return (0.0, 0.0)
    
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_stats.append(statistic_func(sample))
    
    alpha = 1 - confidence
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    return (
        np.percentile(bootstrap_stats, lower_percentile),
        np.percentile(bootstrap_stats, upper_percentile)
    )

def main():
    print("Starting Temperature Analysis Experiment")
    print("=" * 50)
    
    # Setup
    client = setup_openai_client()
    prompts = get_standardized_prompts()
    temperatures = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1]  # 6 conditions
    
    # Target: 60 samples (10 prompts × 6 temperatures)
    # But limit API calls to 30 max, so use 5 prompts × 6 temperatures = 30 calls
    selected_prompts = prompts[:5]
    
    results = []
    api_call_count = 0
    max_api_calls = 30
    
    print(f"Using {len(selected_prompts)} prompts × {len(temperatures)} temperatures = {len(selected_prompts) * len(temperatures)} generations")
    
    start_time = time.time()
    
    for i, prompt in enumerate(selected_prompts):
        for j, temp in enumerate(temperatures):
            if api_call_count >= max_api_calls:
                print(f"Reached maximum API calls ({max_api_calls})")
                break
                
            print(f"Progress: {api_call_count + 1}/{max_api_calls} - Prompt {i+1}, Temp {temp}")
            
            # Generate text
            generated_text = generate_text_with_temperature(client, prompt, temp)
            api_call_count += 1
            
            if generated_text:
                # Calculate metrics
                diversity = calculate_lexical_diversity(generated_text)
                coherence = calculate_semantic_coherence(generated_text)
                word_count = len(re.findall(r'\b\w+\b', generated_text))
                
                results.append({
                    'prompt_id': i,
                    'prompt': prompt,
                    'temperature': temp,
                    'generated_text': generated_text,
                    'lexical_diversity': diversity,
                    'semantic_coherence': coherence,
                    'word_count': word_count
                })
            
            # Small delay to avoid rate limiting
            time.sleep(0.1)
        
        if api_call_count >= max_api_calls:
            break
    
    # Analysis
    print(f"\nAnalysis Results (Total samples: {len(results)})")
    print("=" * 50)
    
    # Group results by temperature
    temp_groups = {}
    for result in results:
        temp = result['temperature']
        if temp not in temp_groups:
            temp_groups[temp] = {
                'coherence': [],
                'diversity': [],
                'word_count': []
            }
        temp_groups[temp]['coherence'].append(result['semantic_coherence'])
        temp_groups[temp]['diversity'].append(result['lexical_diversity'])
        temp_groups[temp]['word_count'].append(result['word_count'])
    
    # Calculate statistics for each temperature
    analysis_results = {}
    for temp in sorted(temp_groups.keys()):
        group = temp_groups[temp]
        
        # Bootstrap confidence intervals
        coherence_ci = bootstrap_confidence_interval(group['coherence'], np.mean, n_bootstrap=500)
        diversity_ci = bootstrap_confidence_interval(group['diversity'], np.mean, n_bootstrap=500)
        
        analysis_results[temp] = {
            'n_samples': len(group['coherence']),
            'coherence_mean': statistics.mean(group['coherence']),
            'coherence_std': statistics.stdev(group['coherence']) if len(group['coherence']) > 1 else 0,
            'coherence_ci': coherence_ci,
            'diversity_mean': statistics.mean(group['diversity']),
            'diversity_std': statistics.stdev(group['diversity']) if len(group['diversity']) > 1 else 0,
            'diversity_ci': diversity_ci,
            'avg_word_count': statistics.mean(group['word_count'])
        }
    
    # Print results table
    print(f"{'Temp':<6} {'N':<3} {'Coherence':<12} {'Diversity':<12} {'Words':<8}")
    print("-" * 50)
    
    for temp in sorted(analysis_results.keys()):
        stats = analysis_results[temp]
        print(f"{temp:<6.1f} {stats['n_samples']:<3} "
              f"{stats['coherence_mean']:<6.3f}±{stats['coherence_std']:<4.3f} "
              f"{stats['diversity_mean']:<6.3f}±{stats['diversity_std']:<4.3f} "
              f"{stats['avg_word_count']:<8.1f}")
    
    # Test hypothesis: coherence decreases with temperature
    temps = sorted(analysis_results.keys())
    coherence_means = [analysis_results[t]['coherence_mean'] for t in temps]
    diversity_means = [analysis_results[t]['diversity_mean'] for t in temps]
    
    # Correlation tests
    coherence_temp_corr = np.corrcoef(temps, coherence_means)[0, 1]
    diversity_temp_corr = np.corrcoef(temps, diversity_means)[0, 1]
    
    print(f"\nCorrelation Analysis:")
    print(f"Temperature vs Coherence: r = {coherence_temp_corr:.3f}")
    print(f"Temperature vs Diversity: r = {diversity_temp_corr:.3f}")
    
    # Find optimal temperature range for diversity
    max_diversity_temp = temps[np.argmax(diversity_means)]
    print(f"Peak diversity at temperature: {max_diversity_temp}")
    
    # Effect sizes (Cohen's d between extreme temperatures)
    if len(temps) >= 2:
        low_temp_coherence = temp_groups[temps[0]]['coherence']
        high_temp_coherence = temp_groups[temps[-1]]['coherence']
        
        if len(low_temp_coherence) > 1 and len(high_temp_coherence) > 1:
            pooled_std = np.sqrt(((len(low_temp_coherence) - 1) * np.var(low_temp_coherence) + 
                                (len(high_temp_coherence) - 1) * np.var(high_temp_coherence)) / 
                               (len(low_temp_coherence) + len(high_temp_coherence) - 2))
            cohens_d = (np.mean(low_temp_coherence) - np.mean(high_temp_coherence)) / pooled_std
            print(f"Cohen's d (low vs high temp coherence): {cohens_d:.3f}")
    
    # Save results
    output_data = {
        'experiment': 'temperature_analysis',
        'timestamp': time.time(),
        'total_samples': len(results),
        'api_calls_used': api_call_count,
        'temperatures_tested': temps,
        'raw_results': results,
        'analysis_summary': analysis_results,
        'correlations': {
            'temperature_coherence': coherence_temp_corr,
            'temperature_diversity': diversity_temp_corr
        },
        'hypothesis_tests': {
            'coherence_decreases_with_temp': coherence_temp_corr < -0.1,
            'diversity_peaks_midrange': 0.3 <= max_diversity_temp <= 0.7
        },
        'runtime_minutes': (time.time() - start_time) / 60
    }
    
    output_path = '/Users/kumacmini/cost-aware-research-search/results/iter_04_results.json'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nExperiment completed in {output_data['runtime_minutes']:.2f} minutes")
    print(f"Results saved to: {output_path}")
    print(f"Total API calls: {api_call_count}")
    
    # Final hypothesis evaluation
    print(f"\nHypothesis Evaluation:")
    print(f"✓ Coherence decreases with temperature: {output_data['hypothesis_tests']['coherence_decreases_with_temp']}")
    print(f"✓ Diversity peaks in mid-range (0.3-0.7): {output_data['hypothesis_tests']['diversity_peaks_midrange']}")

if __name__ == "__main__":
    main()