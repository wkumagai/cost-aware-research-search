import os
import re
import json
import time
import random
import statistics
import numpy as np
import pandas as pd
from scipy import stats
from collections import Counter
import requests
from datetime import datetime

def get_gutenberg_text(book_id):
    """Get text from Project Gutenberg"""
    try:
        url = f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            text = response.text
            # Remove header and footer
            start_markers = ["*** START OF", "***START OF"]
            end_markers = ["*** END OF", "***END OF"]
            
            for marker in start_markers:
                if marker in text:
                    text = text.split(marker, 1)[1]
                    break
            
            for marker in end_markers:
                if marker in text:
                    text = text.split(marker, 1)[0]
                    break
            
            return text.strip()
    except:
        pass
    return None

def get_wikipedia_text(topic):
    """Get Wikipedia article text"""
    try:
        search_url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + topic.replace(" ", "_")
        response = requests.get(search_url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if 'extract' in data and len(data['extract']) > 200:
                return data['extract']
    except:
        pass
    return None

def get_news_text():
    """Get sample news text from mock data"""
    news_samples = [
        "Breaking news today as technology companies announce new initiatives in artificial intelligence development. The recent advances in machine learning have transformed multiple industries, creating opportunities and challenges for businesses worldwide. Experts predict significant changes in employment patterns over the next decade. Companies are investing billions in research and development programs.",
        "Environmental protection agencies report concerning trends in global climate patterns this year. Scientists have documented rising temperatures across multiple regions, affecting agricultural production and wildlife habitats. Conservation efforts continue to expand through international cooperation agreements. Local communities are implementing sustainable practices to reduce environmental impact.",
        "Educational institutions are adapting teaching methods to incorporate digital technologies effectively. Students demonstrate improved engagement when using interactive learning platforms and multimedia resources. Research indicates positive outcomes from personalized learning approaches tailored to individual student needs. Teachers receive professional development training to enhance their technological skills.",
        "Healthcare professionals emphasize the importance of preventive medicine in maintaining public health standards. Recent studies show significant benefits from regular exercise, balanced nutrition, and adequate sleep patterns. Medical advances continue to improve treatment options for various chronic conditions. Community health programs focus on promoting wellness education and early intervention strategies.",
        "Cultural festivals celebrate diverse traditions and artistic expressions throughout the year. Musicians, artists, and performers share their talents with audiences from different backgrounds and communities. These events strengthen social connections and promote understanding between various cultural groups. Local organizations collaborate to preserve historical traditions while embracing contemporary innovations."
    ]
    return random.choice(news_samples)

def call_openai_api(prompt, api_key, max_tokens=400):
    """Call OpenAI API with error handling and rate limiting"""
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    
    data = {
        'model': 'gpt-3.5-turbo',
        'messages': [
            {'role': 'user', 'content': prompt}
        ],
        'max_completion_tokens': max_tokens,
        'temperature': 0.7
    }
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(
                'https://api.openai.com/v1/chat/completions',
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            elif response.status_code == 429:
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                time.sleep(wait_time)
                continue
            else:
                print(f"API error {response.status_code}: {response.text}")
                return None
                
        except Exception as e:
            print(f"API call failed (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            continue
    
    return None

def extract_linguistic_features(text):
    """Extract linguistic features from text"""
    if not text or len(text.strip()) < 50:
        return None
    
    # Clean text
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Extract sentences
    sentence_pattern = r'[.!?]+\s+'
    sentences = [s.strip() for s in re.split(sentence_pattern, text) if s.strip()]
    
    if len(sentences) < 3:
        return None
    
    # Sentence lengths
    sentence_lengths = [len(s.split()) for s in sentences if len(s.split()) > 0]
    
    # Word lengths
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    word_lengths = [len(word) for word in words]
    
    # Punctuation clustering
    punct_pattern = r'[.!?,;:()-]'
    punct_positions = [m.start() for m in re.finditer(punct_pattern, text)]
    
    # Calculate clustering coefficient
    if len(punct_positions) > 2:
        gaps = [punct_positions[i+1] - punct_positions[i] for i in range(len(punct_positions)-1)]
        punct_clustering = statistics.variance(gaps) / (statistics.mean(gaps) ** 2) if statistics.mean(gaps) > 0 else 0
    else:
        punct_clustering = 0
    
    if len(sentence_lengths) < 2 or len(word_lengths) < 10:
        return None
    
    features = {
        'sentence_length_variance': statistics.variance(sentence_lengths),
        'word_length_skewness': stats.skew(word_lengths),
        'punctuation_clustering': punct_clustering,
        'word_count': len(words),
        'sentence_count': len(sentences)
    }
    
    return features

def bootstrap_effect_size(group1, group2, n_bootstrap=500):
    """Calculate bootstrap confidence intervals for effect size"""
    def cohens_d(x, y):
        nx, ny = len(x), len(y)
        if nx < 2 or ny < 2:
            return 0
        pooled_std = np.sqrt(((nx-1)*np.var(x, ddof=1) + (ny-1)*np.var(y, ddof=1)) / (nx+ny-2))
        if pooled_std == 0:
            return 0
        return (np.mean(x) - np.mean(y)) / pooled_std
    
    observed_d = cohens_d(group1, group2)
    
    bootstrap_ds = []
    combined = np.concatenate([group1, group2])
    n1, n2 = len(group1), len(group2)
    
    for _ in range(n_bootstrap):
        resampled = np.random.choice(combined, size=len(combined), replace=True)
        boot_group1 = resampled[:n1]
        boot_group2 = resampled[n1:n1+n2]
        bootstrap_ds.append(cohens_d(boot_group1, boot_group2))
    
    ci_lower = np.percentile(bootstrap_ds, 2.5)
    ci_upper = np.percentile(bootstrap_ds, 97.5)
    
    return observed_d, ci_lower, ci_upper

def main():
    start_time = time.time()
    
    # Get API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    # Data collection
    human_texts = []
    llm_texts = []
    
    topics = ["technology", "environment", "education", "health", "culture"]
    
    print("Collecting human text samples...")
    
    # Gutenberg samples
    gutenberg_ids = [1342, 11, 74, 1661, 2701, 345, 84, 98, 1260, 16, 174, 25344, 5200, 37106]
    gutenberg_count = 0
    for book_id in gutenberg_ids:
        if gutenberg_count >= 15:
            break
        text = get_gutenberg_text(book_id)
        if text and len(text.split()) > 100:
            # Extract excerpt
            words = text.split()
            if len(words) > 500:
                start_idx = random.randint(0, len(words) - 300)
                excerpt = ' '.join(words[start_idx:start_idx + 300])
                human_texts.append({
                    'text': excerpt,
                    'source': 'gutenberg',
                    'topic': 'literature'
                })
                gutenberg_count += 1
        time.sleep(0.5)
    
    # Wikipedia samples
    wiki_topics = ["artificial intelligence", "climate change", "online learning", 
                   "preventive medicine", "cultural diversity", "renewable energy", 
                   "digital education", "public health", "global warming", "machine learning"]
    wiki_count = 0
    for topic in wiki_topics:
        if wiki_count >= 15:
            break
        text = get_wikipedia_text(topic)
        if text and len(text.split()) > 100:
            words = text.split()
            if len(words) > 300:
                excerpt = ' '.join(words[:300])
            else:
                excerpt = text
            human_texts.append({
                'text': excerpt,
                'source': 'wikipedia', 
                'topic': 'encyclopedia'
            })
            wiki_count += 1
        time.sleep(0.5)
    
    # News samples
    for i in range(20):
        text = get_news_text()
        if text and len(text.split()) > 100:
            human_texts.append({
                'text': text,
                'source': 'news',
                'topic': 'current_events'
            })
    
    print(f"Collected {len(human_texts)} human text samples")
    
    # Generate LLM texts
    print("Generating LLM text samples...")
    
    prompts = [
        "Write a 200-300 word informative article about recent advances in artificial intelligence and their impact on society.",
        "Describe environmental challenges facing our planet today and potential solutions being developed by scientists and communities.",
        "Explain how modern educational technology is changing the way students learn and teachers instruct in classrooms worldwide.",
        "Discuss the importance of preventive healthcare measures and how they contribute to maintaining good health throughout life.",
        "Write about cultural festivals and traditions that bring communities together while celebrating diversity and artistic expression.",
        "Analyze the role of renewable energy sources in addressing climate change and creating sustainable economic development.",
        "Examine how digital learning platforms are transforming educational access and opportunities for students globally.",
        "Explore recent medical breakthroughs that are improving treatment options for various health conditions and diseases.",
        "Investigate the impact of social media on modern communication patterns and interpersonal relationships in society.",
        "Discuss sustainable agriculture practices that help farmers increase productivity while protecting environmental resources.",
        "Write about space exploration achievements and their contributions to scientific knowledge and technological innovation.",
        "Explain how urban planning strategies can create more livable, environmentally friendly, and economically vibrant cities.",
        "Analyze the evolution of transportation technology and its effects on commerce, travel, and urban development.",
        "Explore the intersection of art and technology in creating new forms of creative expression and cultural engagement.",
        "Discuss the role of international cooperation in addressing global challenges like poverty, disease, and environmental degradation.",
        "Examine how scientific research methodology ensures reliable knowledge discovery and evidence-based decision making.",
        "Write about the importance of biodiversity conservation and ecosystem protection for environmental sustainability.",
        "Analyze economic trends affecting small businesses and entrepreneurship in the modern global marketplace.",
        "Explore how psychological research informs understanding of human behavior, learning, and mental health treatment.",
        "Discuss innovations in construction technology that improve building safety, efficiency, and environmental sustainability.",
        "Write about the development of clean water technologies and their impact on public health in developing regions.",
        "Examine the role of community organizations in promoting social welfare and civic engagement at local levels.",
        "Analyze how data science and analytics are transforming decision-making processes across various industries.",
        "Explore the relationship between nutrition science and public health policy in promoting healthy lifestyle choices.",
        "Discuss the evolution of communication technology and its influence on global connectivity and information sharing.",
        "Write about archaeological discoveries that provide insights into human history and cultural development patterns.",
        "Examine how financial technology innovations are changing banking services and economic transaction methods.",
        "Analyze the impact of globalization on local cultures while exploring strategies for preserving cultural heritage.",
        "Explore advances in materials science that enable new possibilities in engineering and manufacturing applications.",
        "Discuss the role of libraries and educational institutions in promoting literacy and lifelong learning opportunities."
    ]
    
    api_call_count = 0
    for prompt in prompts:
        if api_call_count >= 30 or len(llm_texts) >= 50:
            break
        
        response = call_openai_api(prompt, api_key, max_tokens=400)
        api_call_count += 1
        
        if response and len(response.split()) > 100:
            llm_texts.append({
                'text': response,
                'source': 'openai',
                'topic': 'generated'
            })
        
        time.sleep(1)  # Rate limiting
    
    print(f"Generated {len(llm_texts)} LLM text samples using {api_call_count} API calls")
    
    # Feature extraction
    print("Extracting linguistic features...")
    
    human_features = []
    for item in human_texts:
        features = extract_linguistic_features(item['text'])
        if features and features['word_count'] >= 50:
            features['source'] = item['source']
            features['topic'] = item['topic']
            human_features.append(features)
    
    llm_features = []
    for item in llm_texts:
        features = extract_linguistic_features(item['text'])
        if features and features['word_count'] >= 50:
            features['source'] = item['source']
            features['topic'] = item['topic']
            llm_features.append(features)
    
    print(f"Extracted features from {len(human_features)} human and {len(llm_features)} LLM documents")
    
    if len(human_features) < 25 or len(llm_features) < 25:
        print(f"Warning: Limited sample sizes - Human: {len(human_features)}, LLM: {len(llm_features)}")
    
    # Statistical analysis
    results = {}
    feature_names = ['sentence_length_variance', 'word_length_skewness', 'punctuation_clustering']
    
    for feature in feature_names:
        human_values = [f[feature] for f in human_features if feature in f and not np.isnan(f[feature])]
        llm_values = [f[feature] for f in llm_features if feature in f and not np.isnan(f[feature])]
        
        if len(human_values) < 10 or len(llm_values) < 10:
            continue
        
        # Remove extreme outliers
        human_values = np.array(human_values)
        llm_values = np.array(llm_values)
        
        # Robust outlier removal
        h_q75, h_q25 = np.percentile(human_values, [75, 25])
        h_iqr = h_q75 - h_q25
        h_mask = (human_values >= h_q25 - 1.5*h_iqr) & (human_values <= h_q75 + 1.5*h_iqr)
        human_values = human_values[h_mask]
        
        l_q75, l_q25 = np.percentile(llm_values, [75, 25])
        l_iqr = l_q75 - l_q25
        l_mask = (llm_values >= l_q25 - 1.5*l_iqr) & (llm_values <= l_q75 + 1.5*l_iqr)
        llm_values = llm_values[l_mask]
        
        # Statistical tests
        try:
            welch_stat, welch_p = stats.ttest_ind(human_values, llm_values, equal_var=False)
            mann_whitney_stat, mann_whitney_p = stats.mannwhitneyu(human_values, llm_values, alternative='two-sided')
            
            # Effect size with bootstrap CI
            cohens_d, ci_lower, ci_upper = bootstrap_effect_size(human_values, llm_values, 500)
            
            # Directional hypothesis testing
            directional_predictions = {
                'sentence_length_variance': 'llm_lower',
                'word_length_skewness': 'llm_lower', 
                'punctuation_clustering': 'llm_higher'
            }
            
            prediction = directional_predictions.get(feature, 'none')
            
            if prediction == 'llm_lower':
                directional_correct = np.mean(llm_values) < np.mean(human_values)
                directional_p = welch_p / 2 if welch_stat > 0 else 1 - welch_p / 2
            elif prediction == 'llm_higher':
                directional_correct = np.mean(llm_values) > np.mean(human_values)
                directional_p = welch_p / 2 if welch_stat < 0 else 1 - welch_p / 2
            else:
                directional_correct = None
                directional_p = welch_p
            
            results[feature] = {
                'human_mean': float(np.mean(human_values)),
                'human_std': float(np.std(human_values)),
                'llm_mean': float(np.mean(llm_values)),
                'llm_std': float(np.std(llm_values)),
                'human_n': int(len(human_values)),
                'llm_n': int(len(llm_values)),
                'welch_t_statistic': float(welch_stat),
                'welch_p_value': float(welch_p),
                'mann_whitney_p': float(mann_whitney_p),
                'cohens_d': float(cohens_d),
                'cohens_d_ci_lower': float(ci_lower),
                'cohens_d_ci_upper': float(ci_upper),
                'effect_size_significant': bool(ci_lower > 0 or ci_upper < 0),
                'predicted_direction': prediction,
                'direction_correct': directional_correct,
                'directional_p_value': float(directional_p)
            }
            
        except Exception as e:
            print(f"Error analyzing {feature}: {e}")
            results[feature] = {'error': str(e)}
    
    # Multiple testing correction (Holm method)
    p_values = [results[f].get('directional_p_value', 1.0) for f in feature_names if f in results and 'directional_p_value' in results[f]]
    if p_values:
        try:
            from scipy.stats import false_discovery_control
            corrected_p = false_discovery_control(p_values, method='holm')
            for i, feature in enumerate([f for f in feature_names if f in results and 'directional_p_value' in results[f]]):
                results[feature]['corrected_p_value'] = float(corrected_p[i])
        except:
            # Fallback manual Holm correction
            sorted_indices = np.argsort(p_values)
            corrected_p = np.zeros(len(p_values))
            for i, idx in enumerate(sorted_indices):
                corrected_p[idx] = min(1.0, p_values[idx] * (len(p_values) - i))
            
            for i, feature in enumerate([f for f in feature_names if f in results and 'directional_p_value' in results[f]]):
                results[feature]['corrected_p_value'] = float(corrected_p[i])
    
    # Topic-stratified analysis
    topic_analysis = {}
    human_topics = list(set([f['topic'] for f in human_features]))
    
    for topic in human_topics:
        topic_human = [f for f in human_features if f['topic'] == topic]
        topic_llm = llm_features  # LLM samples are mixed topics
        
        if len(topic_human) >= 5 and len(topic_llm) >= 5:
            topic_results = {}
            for feature in feature_names:
                h_vals = [f[feature] for f in topic_human if feature in f and not np.isnan(f[feature])]
                l_vals = [f[feature] for f in topic_llm if feature in f and not np.isnan(f[feature])]
                
                if len(h_vals) >= 3 and len(l_vals) >= 3:
                    try:
                        cohens_d, _, _ = bootstrap_effect_size(np.array(h_vals), np.array(l_vals), 100)
                        topic_results[feature] = {
                            'cohens_d': float(cohens_d),
                            'human_mean': float(np.mean(h_vals)),
                            'llm_mean': float(np.mean(l_vals)),
                            'n_human': len(h_vals),
                            'n_llm': len(l_vals)
                        }
                    except:
                        pass
            
            if topic_results:
                topic_analysis[topic] = topic_results
    
    # Summary
    summary = {
        'experiment_info': {
            'timestamp': datetime.now().isoformat(),
            'runtime_seconds': time.time() - start_time,
            'api_calls_used': api_call_count,
            'human_samples_collected': len(human_features),
            'llm_samples_collected': len(llm_features)
        },
        'hypothesis_testing': {
            'directional_predictions': {
                'sentence_variance': 'LLM < Human',
                'word_skewness': 'LLM < Human',
                'punctuation_clustering': 'LLM > Human'
            },
            'success_threshold': 'Effect size > 0.3 with 95% CI excluding zero',
            'multiple_testing_correction': 'Holm method'
        },
        'main_results': results,
        'topic_stratified_results': topic_analysis,
        'robustness_checks': {
            'outlier_removal': 'Applied 1.5 IQR rule',
            'minimum_sample_size': 'Required 10+ samples per group',
            'bootstrap_iterations': 500
        }
    }
    
    # Print summary table
    print("\n" + "="*80)
    print("LINGUISTIC FEATURE DISTRIBUTION ANALYSIS RESULTS")
    print("="*80)
    print(f"Human samples: {len(human_features)} | LLM samples: {len(llm_features)} | API calls: {api_call_count}")
    print()
    print(f"{'Feature':<25} {'Direction':<15} {'Effect Size':<12} {'95% CI':<20} {'P-value':<10} {'Significant'}")
    print("-" * 95)
    
    for feature in feature_names:
        if feature in results and 'cohens_d' in results[feature]:
            r = results[feature]
            direction = "✓" if r.get('direction_correct') else "✗"
            effect_size = f"{r['cohens_d']:.3f}"
            ci = f"[{r['cohens_d_ci_lower']:.3f}, {r['cohens_d_ci_upper']:.3f}]"
            p_val = f"{r.get('corrected_p_value', r['directional_p_value']):.4f}"
            significant = "YES" if r['effect_size_significant'] else "NO"
            
            print(f"{feature:<25} {direction:<15} {effect_size:<12} {ci:<20} {p_val:<10} {significant}")
    
    print("\nHypothesis Support Summary:")
    supported_count = sum(1 for f in feature_names if f in results and results[f].get('effect_size_significant', False))
    print(f"- Features with significant effect sizes (CI excludes 0): {supported_count}/{len(feature_names)}")
    
    correct_direction_count = sum(1 for f in feature_names if f in results and results[f].get('direction_correct', False))
    print(f"- Directional predictions supported: {correct_direction_count}/{len(feature_names)}")
    
    # Save results
    output_path = "/Users/kumacmini/cost-aware-research-search/results/iter_04_results.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    print(f"Total runtime: {time.time() - start_time:.1f} seconds")

if __name__ == "__main__":
    main()