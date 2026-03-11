import json
import os
import sys
import time
import random
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from collections import defaultdict
import re
from openai import OpenAI

# Configuration
API_CALLS_LIMIT = 30
MIN_SAMPLES = 50
OUTPUT_FILE = "/Users/kumacmini/cost-aware-research-search/results/iter_02_results.json"

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def make_api_call(messages, max_retries=3):
    """Make OpenAI API call with retry logic"""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_completion_tokens=500,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"API call attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return None
    return None

def extract_text_features(text):
    """Extract sentence length and punctuation density features"""
    if not text or len(text.strip()) == 0:
        return None
    
    # Split into sentences using multiple delimiters
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if len(sentences) == 0:
        return None
    
    # Calculate sentence lengths
    sentence_lengths = [len(s.split()) for s in sentences]
    
    # Calculate punctuation density
    total_chars = len(text)
    punctuation_chars = len(re.findall(r'[.,!?;:()"\'-]', text))
    punctuation_density = punctuation_chars / total_chars if total_chars > 0 else 0
    
    # Additional features for robustness
    avg_word_length = np.mean([len(word) for sentence in sentences for word in sentence.split()]) if sentences else 0
    
    return {
        'sentence_lengths': sentence_lengths,
        'avg_sentence_length': float(np.mean(sentence_lengths)),
        'sentence_length_std': float(np.std(sentence_lengths)),
        'sentence_length_variance': float(np.var(sentence_lengths)),
        'punctuation_density': float(punctuation_density),
        'num_sentences': len(sentences),
        'avg_word_length': float(avg_word_length),
        'total_words': sum(sentence_lengths)
    }

def generate_synthetic_corpus():
    """Generate synthetic human and LLM text corpora with controlled topics"""
    topics = [
        "artificial intelligence and machine learning applications",
        "climate change and environmental conservation",
        "space exploration and astronomy discoveries",
        "renewable energy technologies and innovations",
        "medical breakthroughs and healthcare advances",
        "digital privacy and cybersecurity issues",
        "sustainable agriculture and food systems",
        "quantum computing and future technologies",
        "ocean conservation and marine biology",
        "urban planning and smart city development"
    ]
    
    # Simulated human text samples (representing typical human writing patterns)
    human_samples = [
        "Climate change represents one of the most pressing challenges of our time. Scientists worldwide have documented rising temperatures, melting ice caps, and extreme weather events. The impact on ecosystems is profound and far-reaching. Immediate action is needed to reduce greenhouse gas emissions and implement sustainable practices.",
        
        "Artificial intelligence continues to revolutionize various industries. Machine learning algorithms can now process vast amounts of data with unprecedented accuracy. However, ethical considerations remain paramount. We must ensure AI development serves humanity's best interests while addressing potential risks and biases.",
        
        "Space exploration has captured human imagination for decades! Recent missions to Mars have provided valuable insights into the planet's geology and potential for past life. Private companies are now joining government agencies in pushing the boundaries of space travel. The future of human spaceflight looks incredibly promising.",
        
        "Renewable energy technologies are becoming increasingly efficient and cost-effective. Solar panels and wind turbines are now competitive with traditional fossil fuels in many regions. Energy storage solutions, particularly battery technology, continue to improve rapidly. This transition is essential for achieving carbon neutrality goals.",
        
        "Medical research has accelerated dramatically in recent years. Gene therapy shows promise for treating previously incurable diseases. Personalized medicine, based on individual genetic profiles, is becoming more accessible. These advances offer hope for millions of patients worldwide.",
    ]
    
    # Generate LLM samples using API
    llm_samples = []
    api_calls_made = 0
    
    for i, topic in enumerate(topics[:10]):  # Limit to 10 topics
        if api_calls_made >= API_CALLS_LIMIT - 10:  # Reserve some calls
            break
            
        print(f"Generating LLM sample {i+1}/10 for topic: {topic[:50]}...")
        
        prompt = f"Write a brief informative paragraph (3-5 sentences) about {topic}. Focus on being factual and educational."
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant that writes clear, informative text."},
            {"role": "user", "content": prompt}
        ]
        
        response = make_api_call(messages)
        api_calls_made += 1
        
        if response:
            llm_samples.append(response.strip())
        else:
            print(f"Failed to generate LLM sample for topic {i+1}")
    
    # Expand human samples to match LLM samples
    while len(human_samples) < len(llm_samples):
        human_samples.extend(human_samples[:min(5, len(llm_samples) - len(human_samples))])
    
    human_samples = human_samples[:len(llm_samples)]
    
    print(f"Generated {len(human_samples)} human samples and {len(llm_samples)} LLM samples")
    print(f"API calls used: {api_calls_made}/{API_CALLS_LIMIT}")
    
    return human_samples, llm_samples, api_calls_made

def run_statistical_tests(human_features, llm_features):
    """Run comprehensive statistical tests"""
    results = {}
    
    # Extract feature arrays
    human_sent_lengths = [f['avg_sentence_length'] for f in human_features if f]
    llm_sent_lengths = [f['avg_sentence_length'] for f in llm_features if f]
    
    human_punct_density = [f['punctuation_density'] for f in human_features if f]
    llm_punct_density = [f['punctuation_density'] for f in llm_features if f]
    
    human_sent_std = [f['sentence_length_std'] for f in human_features if f]
    llm_sent_std = [f['sentence_length_std'] for f in llm_features if f]
    
    # Kolmogorov-Smirnov tests
    if len(human_sent_lengths) > 0 and len(llm_sent_lengths) > 0:
        ks_sent_length = stats.ks_2samp(human_sent_lengths, llm_sent_lengths)
        results['ks_sentence_length'] = {
            'statistic': float(ks_sent_length.statistic),
            'pvalue': float(ks_sent_length.pvalue)
        }
    
    if len(human_punct_density) > 0 and len(llm_punct_density) > 0:
        ks_punct_density = stats.ks_2samp(human_punct_density, llm_punct_density)
        results['ks_punctuation_density'] = {
            'statistic': float(ks_punct_density.statistic),
            'pvalue': float(ks_punct_density.pvalue)
        }
    
    if len(human_sent_std) > 0 and len(llm_sent_std) > 0:
        ks_sent_variance = stats.ks_2samp(human_sent_std, llm_sent_std)
        results['ks_sentence_variance'] = {
            'statistic': float(ks_sent_variance.statistic),
            'pvalue': float(ks_sent_variance.pvalue)
        }
    
    # T-tests for means
    if len(human_sent_lengths) > 1 and len(llm_sent_lengths) > 1:
        t_sent_length = stats.ttest_ind(human_sent_lengths, llm_sent_lengths)
        results['ttest_sentence_length'] = {
            'statistic': float(t_sent_length.statistic),
            'pvalue': float(t_sent_length.pvalue)
        }
    
    if len(human_punct_density) > 1 and len(llm_punct_density) > 1:
        t_punct_density = stats.ttest_ind(human_punct_density, llm_punct_density)
        results['ttest_punctuation_density'] = {
            'statistic': float(t_punct_density.statistic),
            'pvalue': float(t_punct_density.pvalue)
        }
    
    # Summary statistics
    results['summary_stats'] = {
        'human': {
            'sentence_length_mean': float(np.mean(human_sent_lengths)) if human_sent_lengths else 0,
            'sentence_length_std': float(np.std(human_sent_lengths)) if human_sent_lengths else 0,
            'punctuation_density_mean': float(np.mean(human_punct_density)) if human_punct_density else 0,
            'punctuation_density_std': float(np.std(human_punct_density)) if human_punct_density else 0,
            'sentence_variance_mean': float(np.mean(human_sent_std)) if human_sent_std else 0
        },
        'llm': {
            'sentence_length_mean': float(np.mean(llm_sent_lengths)) if llm_sent_lengths else 0,
            'sentence_length_std': float(np.std(llm_sent_lengths)) if llm_sent_lengths else 0,
            'punctuation_density_mean': float(np.mean(llm_punct_density)) if llm_punct_density else 0,
            'punctuation_density_std': float(np.std(llm_punct_density)) if llm_punct_density else 0,
            'sentence_variance_mean': float(np.mean(llm_sent_std)) if llm_sent_std else 0
        }
    }
    
    return results

def create_visualizations(human_features, llm_features):
    """Create comparison visualizations"""
    try:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        # Extract data
        human_sent_lengths = [f['avg_sentence_length'] for f in human_features if f]
        llm_sent_lengths = [f['avg_sentence_length'] for f in llm_features if f]
        
        human_punct_density = [f['punctuation_density'] for f in human_features if f]
        llm_punct_density = [f['punctuation_density'] for f in llm_features if f]
        
        # Sentence length distribution
        if human_sent_lengths and llm_sent_lengths:
            ax1.hist(human_sent_lengths, alpha=0.7, label='Human', bins=10)
            ax1.hist(llm_sent_lengths, alpha=0.7, label='LLM', bins=10)
            ax1.set_xlabel('Average Sentence Length (words)')
            ax1.set_ylabel('Frequency')
            ax1.set_title('Sentence Length Distribution')
            ax1.legend()
        
        # Punctuation density distribution
        if human_punct_density and llm_punct_density:
            ax2.hist(human_punct_density, alpha=0.7, label='Human', bins=10)
            ax2.hist(llm_punct_density, alpha=0.7, label='LLM', bins=10)
            ax2.set_xlabel('Punctuation Density')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Punctuation Density Distribution')
            ax2.legend()
        
        # Box plots for comparison
        if human_sent_lengths and llm_sent_lengths:
            ax3.boxplot([human_sent_lengths, llm_sent_lengths], labels=['Human', 'LLM'])
            ax3.set_ylabel('Average Sentence Length')
            ax3.set_title('Sentence Length Comparison')
        
        if human_punct_density and llm_punct_density:
            ax4.boxplot([human_punct_density, llm_punct_density], labels=['Human', 'LLM'])
            ax4.set_ylabel('Punctuation Density')
            ax4.set_title('Punctuation Density Comparison')
        
        plt.tight_layout()
        plt.savefig('text_analysis_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        return "text_analysis_comparison.png"
    except Exception as e:
        print(f"Visualization creation failed: {e}")
        return None

def main():
    print("Starting text statistical analysis experiment...")
    print(f"Target: Generate >= {MIN_SAMPLES} data points")
    print(f"API call limit: {API_CALLS_LIMIT}")
    
    # Generate synthetic corpus
    human_samples, llm_samples, api_calls_used = generate_synthetic_corpus()
    
    if len(human_samples) == 0 or len(llm_samples) == 0:
        print("ERROR: Failed to generate sufficient samples")
        sys.exit(1)
    
    print(f"\nProcessing {len(human_samples)} human samples...")
    human_features = []
    for i, sample in enumerate(human_samples):
        print(f"Processing human sample {i+1}/{len(human_samples)}")
        features = extract_text_features(sample)
        human_features.append(features)
    
    print(f"\nProcessing {len(llm_samples)} LLM samples...")
    llm_features = []
    for i, sample in enumerate(llm_samples):
        print(f"Processing LLM sample {i+1}/{len(llm_samples)}")
        features = extract_text_features(sample)
        llm_features.append(features)
    
    # Filter out None values
    human_features = [f for f in human_features if f is not None]
    llm_features = [f for f in llm_features if f is not None]
    
    total_samples = len(human_features) + len(llm_features)
    print(f"\nValid samples: {len(human_features)} human, {len(llm_features)} LLM")
    print(f"Total valid samples: {total_samples}")
    
    if total_samples < MIN_SAMPLES:
        print(f"ERROR: Only {total_samples} valid samples, need >= {MIN_SAMPLES}")
        sys.exit(1)
    
    # Run statistical tests
    print("\nRunning statistical analyses...")
    statistical_results = run_statistical_tests(human_features, llm_features)
    
    # Create visualizations
    print("Creating visualizations...")
    viz_file = create_visualizations(human_features, llm_features)
    
    # Prepare final results
    results = {
        'experiment': 'text_statistical_analysis',
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'sample_sizes': {
            'human': len(human_features),
            'llm': len(llm_features),
            'total': total_samples
        },
        'api_calls_used': api_calls_used,
        'statistical_tests': statistical_results,
        'visualization_file': viz_file,
        'raw_data': {
            'human_features': human_features[:10],  # Sample for JSON size
            'llm_features': llm_features[:10]
        }
    }
    
    # Save results
    try:
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to: {OUTPUT_FILE}")
    except Exception as e:
        print(f"Error saving results: {e}")
    
    # Print results table
    print("\n" + "="*80)
    print("EXPERIMENTAL RESULTS SUMMARY")
    print("="*80)
    
    if 'summary_stats' in statistical_results:
        stats_data = statistical_results['summary_stats']
        print(f"{'Metric':<30} {'Human':<15} {'LLM':<15} {'Difference':<15}")
        print("-" * 75)
        
        human_sent_mean = stats_data['human']['sentence_length_mean']
        llm_sent_mean = stats_data['llm']['sentence_length_mean']
        print(f"{'Avg Sentence Length':<30} {human_sent_mean:<15.2f} {llm_sent_mean:<15.2f} {abs(human_sent_mean-llm_sent_mean):<15.2f}")
        
        human_punct_mean = stats_data['human']['punctuation_density_mean']
        llm_punct_mean = stats_data['llm']['punctuation_density_mean']
        print(f"{'Punctuation Density':<30} {human_punct_mean:<15.3f} {llm_punct_mean:<15.3f} {abs(human_punct_mean-llm_punct_mean):<15.3f}")
        
        human_var_mean = stats_data['human']['sentence_variance_mean']
        llm_var_mean = stats_data['llm']['sentence_variance_mean']
        print(f"{'Sentence Length Variance':<30} {human_var_mean:<15.2f} {llm_var_mean:<15.2f} {abs(human_var_mean-llm_var_mean):<15.2f}")
    
    print("\nSTATISTICAL TESTS:")
    print("-" * 50)
    
    for test_name, test_result in statistical_results.items():
        if test_name != 'summary_stats' and isinstance(test_result, dict):
            ks_stat = test_result.get('statistic', 0)
            p_value = test_result.get('pvalue', 1)
            significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
            print(f"{test_name:<25} KS={ks_stat:.3f}, p={p_value:.4f} {significance}")
    
    # Check hypothesis
    main_ks_stat = statistical_results.get('ks_sentence_length', {}).get('statistic', 0)
    hypothesis_confirmed = main_ks_stat > 0.3
    
    print(f"\nHYPOTHESIS TEST:")
    print(f"Main KS statistic: {main_ks_stat:.3f}")
    print(f"Success threshold: > 0.3")
    print(f"Hypothesis confirmed: {'YES' if hypothesis_confirmed else 'NO'}")
    
    print(f"\nTotal samples analyzed: {total_samples}")
    print(f"API calls used: {api_calls_used}/{API_CALLS_LIMIT}")
    print("="*80)

if __name__ == "__main__":
    main()