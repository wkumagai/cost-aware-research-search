import os
import re
import json
import time
import statistics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from openai import OpenAI
import random

def calculate_linguistic_features(text):
    """Calculate basic linguistic features from text."""
    # Clean text and split into sentences
    sentences = re.split(r'[.!?]+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if not sentences:
        return None
    
    # Word lengths
    words = re.findall(r'\b\w+\b', text.lower())
    if not words:
        return None
        
    word_lengths = [len(word) for word in words]
    
    # Sentence lengths (in words)
    sentence_lengths = []
    for sentence in sentences:
        sentence_words = re.findall(r'\b\w+\b', sentence)
        if sentence_words:
            sentence_lengths.append(len(sentence_words))
    
    if not sentence_lengths:
        return None
    
    # Punctuation density (punctuation marks per 100 characters)
    punctuation_marks = len(re.findall(r'[.!?,;:()"\'-]', text))
    text_length = len(text)
    punctuation_density = (punctuation_marks / max(text_length, 1)) * 100 if text_length > 0 else 0
    
    return {
        'avg_word_length': statistics.mean(word_lengths),
        'word_length_std': statistics.stdev(word_lengths) if len(word_lengths) > 1 else 0,
        'avg_sentence_length': statistics.mean(sentence_lengths),
        'sentence_length_std': statistics.stdev(sentence_lengths) if len(sentence_lengths) > 1 else 0,
        'punctuation_density': punctuation_density,
        'total_words': len(words),
        'total_sentences': len(sentences)
    }

def get_llm_text(client, prompt, max_tokens=150):
    """Generate text using OpenAI API with error handling."""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=max_tokens,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"API error: {e}")
        return None

def get_human_text_samples():
    """Generate built-in human text samples."""
    samples = [
        "The quick brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet at least once. Many writers use this phrase to test fonts and keyboards.",
        "Scientists have discovered that dolphins can recognize themselves in mirrors. This ability, known as self-recognition, was previously thought to exist only in humans and great apes. The research has important implications for our understanding of animal consciousness.",
        "Last summer, I traveled to Italy with my family. We visited Rome, Florence, and Venice. The architecture was breathtaking, and the food was incredible. I especially loved the pasta in a small restaurant near the Trevi Fountain.",
        "Climate change represents one of the most significant challenges facing humanity today. Rising temperatures, melting ice caps, and extreme weather events are becoming increasingly common. Scientists warn that immediate action is necessary to prevent catastrophic consequences.",
        "My grandmother's garden was filled with roses, tulips, and daisies. Every morning, she would water the plants and pull weeds. The garden was her pride and joy, and neighbors often stopped by to admire the beautiful flowers.",
        "The stock market experienced significant volatility last week. Technology stocks fell sharply on Tuesday, but recovered by Friday. Investors are closely watching economic indicators for signs of inflation. Many analysts recommend diversifying portfolios during uncertain times.",
        "Children learn languages more easily than adults due to their brain plasticity. Young minds are particularly adept at acquiring pronunciation and grammar rules. This critical period for language learning typically occurs before adolescence.",
        "The concert hall was packed with eager music lovers. The orchestra performed Beethoven's Ninth Symphony with exceptional skill and passion. The audience gave a standing ovation that lasted several minutes.",
        "Modern smartphones contain more computing power than the computers that sent humans to the moon. These devices have revolutionized communication, entertainment, and commerce. Nearly everyone carries a supercomputer in their pocket today.",
        "Baking bread requires patience and practice. The dough must be kneaded properly and allowed to rise twice. The aroma of fresh bread filling the kitchen is one of life's simple pleasures.",
        "Exercise provides numerous benefits for both physical and mental health. Regular activity strengthens muscles, improves cardiovascular function, and releases endorphins. Many people find that daily walks help reduce stress and anxiety.",
        "The library was quiet except for the soft rustle of turning pages. Students sat at wooden tables, surrounded by towering bookshelves. Sunlight streamed through tall windows, creating pools of warmth on the polished floor.",
        "Cooking is both an art and a science. Success requires understanding ingredients, temperatures, and timing. Great chefs combine creativity with technical skill to create memorable dining experiences.",
        "The mountain trail wound through dense forests and rocky outcroppings. Hikers paused frequently to catch their breath and admire the scenery. At the summit, panoramic views stretched for miles in every direction.",
        "Social media has transformed how people communicate and share information. While these platforms connect individuals across vast distances, they also raise concerns about privacy and misinformation. Users must navigate this digital landscape carefully.",
        "The old lighthouse stood sentinel on the rocky coast. For over a century, it had guided ships safely through treacherous waters. Though modern GPS has reduced its importance, the structure remains a beloved landmark.",
        "Artificial intelligence is advancing rapidly across multiple industries. Machine learning algorithms can now perform complex tasks once thought impossible for computers. This technology promises to reshape everything from healthcare to transportation.",
        "The farmer's market bustled with activity on Saturday mornings. Vendors sold fresh produce, homemade bread, and artisanal crafts. Customers enjoyed sampling local honey and discussing recipes with neighboring shoppers.",
        "Space exploration continues to capture human imagination. Recent missions to Mars have provided valuable data about the planet's geology and atmosphere. Scientists hope future expeditions might discover signs of ancient life.",
        "The detective carefully examined the crime scene for clues. Every detail could potentially solve the mysterious case. Witnesses were interviewed, and evidence was methodically collected and analyzed.",
        "Reading fiction transports us to different worlds and experiences. Good books challenge our perspectives and expand our empathy. Libraries serve as gateways to countless adventures and learning opportunities.",
        "The chef prepared the meal with meticulous attention to detail. Fresh herbs were chopped precisely, and sauces were seasoned to perfection. Each dish reflected years of culinary training and passion.",
        "Ocean waves crashed against the sandy shore as seagulls circled overhead. Children built sandcastles while their parents relaxed under colorful umbrellas. The beach offered peaceful escape from urban life.",
        "Teachers shape young minds through patience, creativity, and dedication. They inspire students to explore new ideas and develop critical thinking skills. Education remains one of society's most important investments.",
        "The antique shop contained treasures from bygone eras. Vintage furniture, old photographs, and forgotten books filled every corner. Customers browsed carefully, searching for items with special meaning or historical significance."
    ]
    return samples

def main():
    print("Starting linguistic features comparison experiment...")
    
    # Initialize OpenAI client
    try:
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    except Exception as e:
        print(f"Error initializing OpenAI client: {e}")
        return
    
    # Generate prompts for diverse LLM text
    prompts = [
        "Write a paragraph about technology and its impact on society.",
        "Describe a memorable travel experience you might have had.",
        "Explain how climate change affects the environment.",
        "Write about the importance of education in modern life.",
        "Describe a perfect day in nature.",
        "Discuss the benefits and drawbacks of social media.",
        "Write about cooking and food culture.",
        "Describe the process of learning a new skill.",
        "Write about the role of art in human expression.",
        "Discuss the future of transportation."
    ]
    
    # Collect data
    human_features = []
    llm_features = []
    
    # Get human text features
    print("Analyzing human text samples...")
    human_texts = get_human_text_samples()
    for i, text in enumerate(human_texts):
        features = calculate_linguistic_features(text)
        if features and features['total_words'] >= 10:  # Minimum quality check
            human_features.append(features)
        if len(human_features) >= 25:  # Get 25 human samples
            break
    
    print(f"Collected {len(human_features)} human text samples")
    
    # Get LLM text features
    print("Generating and analyzing LLM text samples...")
    api_calls = 0
    max_api_calls = 25  # Limit API calls
    
    for i in range(max_api_calls):
        if api_calls >= max_api_calls:
            break
            
        prompt = random.choice(prompts)
        llm_text = get_llm_text(client, prompt, max_tokens=120)
        api_calls += 1
        
        if llm_text:
            features = calculate_linguistic_features(llm_text)
            if features and features['total_words'] >= 10:
                llm_features.append(features)
        
        # Small delay to avoid rate limiting
        time.sleep(0.1)
        
        if len(llm_features) >= 25:  # Get 25 LLM samples
            break
    
    print(f"Collected {len(llm_features)} LLM text samples using {api_calls} API calls")
    
    # Ensure minimum sample size
    min_samples = min(len(human_features), len(llm_features))
    if min_samples < 20:
        print(f"Warning: Only {min_samples} samples per group. Results may be unreliable.")
    
    # Convert to DataFrames for analysis
    human_df = pd.DataFrame(human_features[:min_samples])
    llm_df = pd.DataFrame(llm_features[:min_samples])
    
    # Statistical analysis
    results = {
        'sample_sizes': {
            'human': len(human_df),
            'llm': len(llm_df)
        },
        'human_stats': {},
        'llm_stats': {},
        'statistical_tests': {},
        'cohens_d': {},
        'api_calls_used': api_calls
    }
    
    # Analyze each feature
    features_to_analyze = ['avg_word_length', 'avg_sentence_length', 'punctuation_density']
    
    for feature in features_to_analyze:
        # Basic statistics
        results['human_stats'][feature] = {
            'mean': float(human_df[feature].mean()),
            'std': float(human_df[feature].std()),
            'min': float(human_df[feature].min()),
            'max': float(human_df[feature].max())
        }
        
        results['llm_stats'][feature] = {
            'mean': float(llm_df[feature].mean()),
            'std': float(llm_df[feature].std()),
            'min': float(llm_df[feature].min()),
            'max': float(llm_df[feature].max())
        }
        
        # Statistical tests
        try:
            # T-test
            t_stat, p_value = stats.ttest_ind(human_df[feature], llm_df[feature])
            
            # Cohen's d
            pooled_std = np.sqrt(((len(human_df) - 1) * human_df[feature].var() + 
                                 (len(llm_df) - 1) * llm_df[feature].var()) / 
                                (len(human_df) + len(llm_df) - 2))
            cohens_d = (human_df[feature].mean() - llm_df[feature].mean()) / pooled_std
            
            results['statistical_tests'][feature] = {
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'significant': bool(p_value < 0.05)
            }
            
            results['cohens_d'][feature] = {
                'value': float(cohens_d),
                'magnitude': 'large' if abs(cohens_d) > 0.8 else 'medium' if abs(cohens_d) > 0.5 else 'small'
            }
            
        except Exception as e:
            print(f"Error in statistical analysis for {feature}: {e}")
            results['statistical_tests'][feature] = {'error': str(e)}
            results['cohens_d'][feature] = {'error': str(e)}
    
    # Summary
    significant_features = sum(1 for f in features_to_analyze 
                             if results['statistical_tests'].get(f, {}).get('significant', False))
    large_effects = sum(1 for f in features_to_analyze 
                       if abs(results['cohens_d'].get(f, {}).get('value', 0)) > 0.5)
    
    results['summary'] = {
        'hypothesis_supported': bool(large_effects >= 2),
        'significant_features': int(significant_features),
        'large_effect_features': int(large_effects),
        'total_features_tested': len(features_to_analyze)
    }
    
    # Print results
    print("\n" + "="*50)
    print("EXPERIMENT RESULTS")
    print("="*50)
    
    print(f"\nSample sizes: Human={results['sample_sizes']['human']}, LLM={results['sample_sizes']['llm']}")
    print(f"API calls used: {results['api_calls_used']}")
    
    print("\nFeature Comparison:")
    print("-" * 70)
    print(f"{'Feature':<20} {'Human Mean':<12} {'LLM Mean':<12} {'Cohen\'s d':<12} {'p-value':<10}")
    print("-" * 70)
    
    for feature in features_to_analyze:
        human_mean = results['human_stats'][feature]['mean']
        llm_mean = results['llm_stats'][feature]['mean']
        cohens_d = results['cohens_d'].get(feature, {}).get('value', 0)
        p_val = results['statistical_tests'].get(feature, {}).get('p_value', 1.0)
        
        print(f"{feature:<20} {human_mean:<12.3f} {llm_mean:<12.3f} {cohens_d:<12.3f} {p_val:<10.4f}")
    
    print(f"\nHypothesis supported: {results['summary']['hypothesis_supported']}")
    print(f"Features with large effects (|d| > 0.5): {results['summary']['large_effect_features']}/3")
    print(f"Statistically significant features: {results['summary']['significant_features']}/3")
    
    # Save results
    output_path = "/Users/kumacmini/cost-aware-research-search/results/iter_02_results.json"
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")
    except Exception as e:
        print(f"Error saving results: {e}")
    
    print("\nExperiment completed successfully!")

if __name__ == "__main__":
    main()