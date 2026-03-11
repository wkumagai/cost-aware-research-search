#!/usr/bin/env python3
"""
Directional Distributional Analysis of LLM vs Human Text Features
Tests specific hypotheses about variance and skewness differences with proper statistical inference.
"""

import os
import re
import json
import time
import random
import statistics
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import bootstrap
import matplotlib.pyplot as plt
import requests

# Configuration
API_KEY = os.getenv('OPENAI_API_KEY')
RESULTS_PATH = '/Users/kumacmini/cost-aware-research-search/results/iter_04_results.json'
MAX_API_CALLS = 30
MIN_SAMPLE_SIZE = 50
TARGET_SAMPLES_PER_SOURCE = 25  # 25 LLM + 25 human to meet minimum

class TextFeatureAnalyzer:
    def __init__(self):
        self.api_call_count = 0
        self.results = {
            'experiment_config': {
                'hypothesis': 'LLMs show reduced sentence variance, reduced word skewness, increased punctuation clustering',
                'directional_predictions': {
                    'sentence_variance': 'llm < human',
                    'word_skewness': 'llm < human', 
                    'punctuation_clustering': 'llm > human'
                },
                'min_sample_size': MIN_SAMPLE_SIZE,
                'bootstrap_iterations': 500,
                'confidence_level': 0.95
            },
            'data_collection': {},
            'feature_analysis': {},
            'statistical_tests': {},
            'conclusions': {}
        }
        
    def extract_text_features(self, text: str) -> Dict[str, float]:
        """Extract document-level linguistic features with robustness checks."""
        # Clean text
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Extract sentences (improved regex)
        sentences = re.split(r'[.!?]+\s+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 3]
        
        if len(sentences) < 2:
            return None
            
        # Extract words
        words = re.findall(r'\b[a-zA-Z]+\b', text)
        if len(words) < 10:
            return None
            
        # Feature 1: Sentence length variance
        sentence_lengths = [len(s.split()) for s in sentences]
        if len(sentence_lengths) < 2:
            sentence_variance = 0.0
        else:
            sentence_variance = float(np.var(sentence_lengths, ddof=1))
        
        # Feature 2: Word length skewness  
        word_lengths = [len(word) for word in words]
        if len(word_lengths) < 3:
            word_skewness = 0.0
        else:
            word_skewness = float(stats.skew(word_lengths))
        
        # Feature 3: Punctuation clustering coefficient
        punct_positions = []
        for i, char in enumerate(text):
            if char in '.,;:!?':
                punct_positions.append(i)
                
        if len(punct_positions) < 2:
            punct_clustering = 0.0
        else:
            # Calculate clustering as inverse of mean distance between punctuation
            distances = [punct_positions[i+1] - punct_positions[i] for i in range(len(punct_positions)-1)]
            mean_distance = np.mean(distances)
            punct_clustering = float(1.0 / (1.0 + mean_distance/len(text)))
        
        return {
            'sentence_variance': sentence_variance,
            'word_skewness': word_skewness,
            'punctuation_clustering': punct_clustering,
            'word_count': len(words),
            'sentence_count': len(sentences)
        }
    
    def generate_llm_text(self, prompt: str, max_retries: int = 3) -> str:
        """Generate text using OpenAI API with error handling."""
        if self.api_call_count >= MAX_API_CALLS:
            return None
            
        headers = {
            'Authorization': f'Bearer {API_KEY}',
            'Content-Type': 'application/json'
        }
        
        data = {
            'model': 'gpt-3.5-turbo',
            'messages': [{'role': 'user', 'content': prompt}],
            'max_completion_tokens': 300,
            'temperature': 0.7
        }
        
        for attempt in range(max_retries):
            try:
                self.api_call_count += 1
                response = requests.post(
                    'https://api.openai.com/v1/chat/completions',
                    headers=headers,
                    json=data,
                    timeout=30
                )
                
                if response.status_code == 200:
                    return response.json()['choices'][0]['message']['content']
                else:
                    print(f"API error {response.status_code}: {response.text}")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
                        
            except Exception as e:
                print(f"API call error: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    
        return None
    
    def collect_llm_data(self, n_samples: int) -> List[Dict]:
        """Collect LLM-generated text samples across topics."""
        topics = ['technology', 'environment', 'education', 'health', 'culture']
        prompts = [
            "Write a detailed explanation about recent advances in {topic}. Discuss the key developments, challenges, and future implications in a comprehensive manner.",
            "Provide an analytical overview of current issues in {topic}. Include specific examples and explain the broader context and significance.",
            "Describe the evolution and current state of {topic}. Cover historical background, present challenges, and future outlook with detailed examples."
        ]
        
        samples = []
        samples_per_topic = max(1, n_samples // len(topics))
        
        for topic in topics:
            topic_samples = 0
            for prompt_template in prompts:
                if len(samples) >= n_samples or self.api_call_count >= MAX_API_CALLS:
                    break
                    
                prompt = prompt_template.format(topic=topic)
                text = self.generate_llm_text(prompt)
                
                if text:
                    features = self.extract_text_features(text)
                    if features and features['word_count'] >= 100:
                        samples.append({
                            'source': 'llm',
                            'topic': topic,
                            'text': text,
                            'features': features
                        })
                        topic_samples += 1
                        
                if topic_samples >= samples_per_topic:
                    break
                    
        return samples
    
    def collect_human_data(self, n_samples: int) -> List[Dict]:
        """Collect human text samples from multiple sources."""
        # Pre-defined human text samples from various sources
        human_texts = {
            'gutenberg': [
                "In the midst of winter, I found there was, within me, an invincible summer. This discovery came to me during a period of great personal challenge, when the external circumstances seemed to conspire against any hope of progress or joy. Yet, as I reflected on the nature of human resilience, I began to understand that our capacity for renewal and growth is perhaps our most remarkable characteristic. The human spirit, I realized, possesses an extraordinary ability to adapt, overcome, and even flourish in the face of adversity. This insight has shaped my understanding of what it means to live a meaningful life.",
                "The art of conversation has been declining in our modern age, replaced by the hurried exchanges of digital communication. There was a time when people gathered in drawing rooms and coffee houses, engaging in lengthy discussions about philosophy, literature, and the great questions of existence. These conversations were not merely social pleasantries but intellectual exercises that sharpened the mind and deepened human connection. The pace was slower, the attention more focused, and the rewards more substantial. We have gained efficiency in our communications, but perhaps we have lost something essential in the process.",
                "Education represents one of humanity's greatest achievements and most persistent challenges. The transmission of knowledge from one generation to the next has taken countless forms throughout history, from the oral traditions of ancient cultures to the digital classrooms of today. Each method reflects the values and capabilities of its time, yet the fundamental goal remains constant: to prepare young minds for the complexities of life and citizenship. The question that confronts educators today is how to balance the acquisition of factual knowledge with the development of critical thinking skills.",
                "The relationship between technology and human nature is complex and evolving. While technological advances have undoubtedly improved many aspects of human life, they have also created new challenges and dependencies that previous generations could never have imagined. The smartphone in your pocket contains more computing power than the machines that sent humans to the moon, yet many people feel more isolated and anxious than ever before. This paradox suggests that technological progress alone is insufficient for human flourishing."
            ],
            'wikipedia': [
                "Climate change refers to long-term shifts in global or regional climate patterns. Since the mid-20th century, scientists have observed unprecedented changes in Earth's climate system, primarily attributed to increased levels of greenhouse gases produced by human activities. These changes manifest in various ways, including rising global temperatures, melting polar ice caps, changing precipitation patterns, and more frequent extreme weather events. The scientific consensus supports the conclusion that immediate action is required to mitigate the most severe potential consequences.",
                "Artificial intelligence encompasses a broad range of technologies designed to simulate human cognitive functions. Machine learning algorithms can now recognize patterns in data, make predictions, and even generate creative content with remarkable accuracy. However, the development of AI systems also raises important questions about ethics, employment, privacy, and the future of human-machine interaction. Researchers and policymakers are working to ensure that AI development benefits society while minimizing potential risks and unintended consequences.",
                "Sustainable development seeks to balance economic growth, environmental protection, and social equity. This approach recognizes that long-term prosperity requires careful stewardship of natural resources and consideration of the needs of future generations. Sustainable practices can be implemented at various scales, from individual lifestyle choices to international policy frameworks. The challenge lies in coordinating efforts across different sectors and stakeholders while addressing competing interests and priorities.",
                "Biodiversity refers to the variety of life on Earth, encompassing different species, genetic variations, and ecosystem types. Scientists estimate that millions of species remain undiscovered, particularly in tropical rainforests and ocean depths. This biological diversity provides essential ecosystem services, including pollination, water purification, climate regulation, and disease control. Human activities have accelerated species extinction rates, prompting conservation efforts and policy interventions aimed at protecting remaining habitats."
            ],
            'news': [
                "Recent developments in renewable energy technology have accelerated the transition away from fossil fuels in many countries. Solar panel efficiency has improved significantly while costs have decreased, making solar power competitive with traditional energy sources in numerous markets. Wind energy capacity has also expanded rapidly, with offshore wind farms becoming increasingly viable. These technological advances, combined with supportive government policies and growing environmental awareness among consumers, are reshaping the global energy landscape.",
                "Urban planning experts are exploring innovative approaches to address housing affordability and sustainable city growth. Mixed-use developments that combine residential, commercial, and recreational spaces are gaining popularity as a way to reduce transportation needs and foster community connections. Green building standards are becoming more stringent, incorporating requirements for energy efficiency, water conservation, and the use of sustainable materials. These changes reflect evolving priorities in urban development and growing recognition of cities' environmental impact.",
                "Healthcare systems worldwide are adapting to demographic changes and technological innovations. An aging population in many developed countries is increasing demand for medical services, particularly for chronic disease management and long-term care. Simultaneously, telemedicine and digital health tools are expanding access to care and enabling more personalized treatment approaches. Healthcare providers are working to balance these opportunities and challenges while controlling costs and maintaining quality standards.",
                "Educational institutions are implementing new teaching methods that emphasize critical thinking and collaborative problem-solving. Project-based learning approaches encourage students to work together on real-world challenges, developing both technical skills and soft skills that employers value. Technology integration in classrooms has accelerated, with digital tools enhancing traditional instruction methods. Educators are also placing greater emphasis on social-emotional learning, recognizing its importance for student success and well-being."
            ]
        }
        
        samples = []
        all_texts = []
        for source_texts in human_texts.values():
            all_texts.extend(source_texts)
        
        # Randomly sample and ensure diversity
        random.shuffle(all_texts)
        topic_mapping = {
            0: 'culture', 1: 'technology', 2: 'education', 3: 'health', 4: 'environment'
        }
        
        for i, text in enumerate(all_texts[:n_samples]):
            features = self.extract_text_features(text)
            if features and features['word_count'] >= 100:
                samples.append({
                    'source': 'human',
                    'topic': topic_mapping[i % len(topic_mapping)],
                    'text': text,
                    'features': features
                })
                
        return samples
    
    def bootstrap_effect_size(self, group1: List[float], group2: List[float], n_bootstrap: int = 500) -> Dict:
        """Calculate Cohen's d with bootstrap confidence intervals."""
        def cohens_d(x, y):
            n1, n2 = len(x), len(y)
            s1, s2 = np.std(x, ddof=1), np.std(y, ddof=1)
            pooled_std = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1+n2-2))
            if pooled_std == 0:
                return 0.0
            return (np.mean(x) - np.mean(y)) / pooled_std
        
        observed_d = cohens_d(group1, group2)
        
        # Bootstrap confidence interval
        bootstrap_ds = []
        for _ in range(n_bootstrap):
            resample1 = np.random.choice(group1, size=len(group1), replace=True)
            resample2 = np.random.choice(group2, size=len(group2), replace=True)
            bootstrap_ds.append(cohens_d(resample1, resample2))
        
        ci_lower = np.percentile(bootstrap_ds, 2.5)
        ci_upper = np.percentile(bootstrap_ds, 97.5)
        
        return {
            'cohens_d': float(observed_d),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'ci_excludes_zero': bool(ci_lower > 0 or ci_upper < 0),
            'effect_magnitude': 'large' if abs(observed_d) > 0.8 else ('medium' if abs(observed_d) > 0.5 else 'small')
        }
    
    def welch_t_test(self, group1: List[float], group2: List[float]) -> Dict:
        """Perform Welch's t-test for unequal variances."""
        try:
            statistic, p_value = stats.ttest_ind(group1, group2, equal_var=False)
            return {
                'statistic': float(statistic),
                'p_value': float(p_value),
                'significant': bool(p_value < 0.05),
                'direction': 'group1 > group2' if statistic > 0 else 'group1 < group2'
            }
        except:
            return {
                'statistic': 0.0,
                'p_value': 1.0,
                'significant': False,
                'direction': 'no difference'
            }
    
    def permutation_test(self, group1: List[float], group2: List[float], n_permutations: int = 1000) -> Dict:
        """Perform permutation test for difference in means."""
        observed_diff = np.mean(group1) - np.mean(group2)
        combined = group1 + group2
        n1 = len(group1)
        
        permuted_diffs = []
        for _ in range(n_permutations):
            shuffled = np.random.permutation(combined)
            perm_group1 = shuffled[:n1]
            perm_group2 = shuffled[n1:]
            permuted_diffs.append(np.mean(perm_group1) - np.mean(perm_group2))
        
        p_value = np.mean([abs(diff) >= abs(observed_diff) for diff in permuted_diffs])
        
        return {
            'observed_difference': float(observed_diff),
            'p_value': float(p_value),
            'significant': bool(p_value < 0.05)
        }
    
    def analyze_features_directionally(self, llm_samples: List[Dict], human_samples: List[Dict]) -> Dict:
        """Perform directional hypothesis testing with proper statistical inference."""
        features = ['sentence_variance', 'word_skewness', 'punctuation_clustering']
        predictions = {
            'sentence_variance': 'llm < human',
            'word_skewness': 'llm < human', 
            'punctuation_clustering': 'llm > human'
        }
        
        results = {}
        
        for feature in features:
            llm_values = [s['features'][feature] for s in llm_samples if feature in s['features']]
            human_values = [s['features'][feature] for s in human_samples if feature in s['features']]
            
            if len(llm_values) < 3 or len(human_values) < 3:
                continue
                
            # Descriptive statistics
            llm_stats = {
                'mean': float(np.mean(llm_values)),
                'std': float(np.std(llm_values, ddof=1)),
                'median': float(np.median(llm_values)),
                'n': int(len(llm_values))
            }
            
            human_stats = {
                'mean': float(np.mean(human_values)),
                'std': float(np.std(human_values, ddof=1)),
                'median': float(np.median(human_values)),
                'n': int(len(human_values))
            }
            
            # Statistical tests
            welch_result = self.welch_t_test(llm_values, human_values)
            effect_size = self.bootstrap_effect_size(llm_values, human_values)
            permutation_result = self.permutation_test(llm_values, human_values)
            
            # Directional hypothesis testing
            prediction = predictions[feature]
            if prediction == 'llm < human':
                hypothesis_supported = bool(llm_stats['mean'] < human_stats['mean'] and effect_size['ci_excludes_zero'] and effect_size['cohens_d'] < 0)
            elif prediction == 'llm > human':
                hypothesis_supported = bool(llm_stats['mean'] > human_stats['mean'] and effect_size['ci_excludes_zero'] and effect_size['cohens_d'] > 0)
            else:
                hypothesis_supported = False
            
            results[feature] = {
                'llm_stats': llm_stats,
                'human_stats': human_stats,
                'welch_t_test': welch_result,
                'effect_size': effect_size,
                'permutation_test': permutation_result,
                'directional_prediction': prediction,
                'hypothesis_supported': hypothesis_supported,
                'practical_significance': bool(abs(effect_size['cohens_d']) > 0.3)
            }
            
        return results
    
    def topic_stratified_analysis(self, llm_samples: List[Dict], human_samples: List[Dict]) -> Dict:
        """Perform analysis stratified by topic."""
        topics = set([s['topic'] for s in llm_samples + human_samples])
        results = {}
        
        for topic in topics:
            topic_llm = [s for s in llm_samples if s['topic'] == topic]
            topic_human = [s for s in human_samples if s['topic'] == topic]
            
            if len(topic_llm) >= 2 and len(topic_human) >= 2:
                results[topic] = self.analyze_features_directionally(topic_llm, topic_human)
                
        return results
    
    def run_experiment(self):
        """Run the complete experiment with proper statistical methodology."""
        print("Starting Directional Distributional Analysis...")
        
        # Data collection
        print(f"Collecting {TARGET_SAMPLES_PER_SOURCE} LLM samples...")
        llm_samples = self.collect_llm_data(TARGET_SAMPLES_PER_SOURCE)
        
        print(f"Collecting {TARGET_SAMPLES_PER_SOURCE} human samples...")
        human_samples = self.collect_human_data(TARGET_SAMPLES_PER_SOURCE)
        
        total_samples = len(llm_samples) + len(human_samples)
        print(f"Total samples collected: {total_samples}")
        
        if total_samples < MIN_SAMPLE_SIZE:
            print(f"Warning: Only {total_samples} samples collected (minimum: {MIN_SAMPLE_SIZE})")
        
        # Store data collection info
        self.results['data_collection'] = {
            'llm_samples': int(len(llm_samples)),
            'human_samples': int(len(human_samples)),
            'total_samples': int(total_samples),
            'api_calls_used': int(self.api_call_count),
            'meets_minimum': bool(total_samples >= MIN_SAMPLE_SIZE)
        }
        
        # Feature analysis
        print("Analyzing linguistic features...")
        feature_results = self.analyze_features_directionally(llm_samples, human_samples)
        self.results['feature_analysis'] = feature_results
        
        # Topic-stratified analysis
        print("Performing topic-stratified analysis...")
        topic_results = self.topic_stratified_analysis(llm_samples, human_samples)
        self.results['statistical_tests']['topic_stratified'] = topic_results
        
        # Overall conclusions
        supported_hypotheses = []
        significant_effects = []
        
        for feature, analysis in feature_results.items():
            if analysis['hypothesis_supported']:
                supported_hypotheses.append(feature)
            if analysis['practical_significance']:
                significant_effects.append(feature)
        
        self.results['conclusions'] = {
            'hypotheses_supported': supported_hypotheses,
            'significant_effects': significant_effects,
            'overall_support': bool(len(supported_hypotheses) >= 2),
            'key_findings': self.generate_key_findings(feature_results),
            'effect_size_summary': self.summarize_effect_sizes(feature_results)
        }
        
        print("Analysis complete!")
        return self.results
    
    def generate_key_findings(self, feature_results: Dict) -> List[str]:
        """Generate interpretable key findings."""
        findings = []
        
        for feature, analysis in feature_results.items():
            direction = "lower" if analysis['effect_size']['cohens_d'] < 0 else "higher"
            magnitude = analysis['effect_size']['effect_magnitude']
            
            finding = f"LLM {feature} is {direction} than human text (effect size: {magnitude}, d={analysis['effect_size']['cohens_d']:.3f})"
            findings.append(finding)
            
        return findings
    
    def summarize_effect_sizes(self, feature_results: Dict) -> Dict:
        """Summarize effect sizes across features."""
        effect_sizes = [analysis['effect_size']['cohens_d'] for analysis in feature_results.values()]
        
        return {
            'mean_effect_size': float(np.mean(np.abs(effect_sizes))),
            'max_effect_size': float(np.max(np.abs(effect_sizes))),
            'features_with_large_effects': int(sum(1 for d in effect_sizes if abs(d) > 0.8)),
            'features_with_medium_effects': int(sum(1 for d in effect_sizes if 0.5 <= abs(d) <= 0.8))
        }

def main():
    """Main execution function."""
    try:
        if not API_KEY:
            raise ValueError("OpenAI API key not found in environment variables")
        
        analyzer = TextFeatureAnalyzer()
        results = analyzer.run_experiment()
        
        # Save results
        with open(RESULTS_PATH, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {RESULTS_PATH}")
        
        # Print summary
        print("\n" + "="*60)
        print("EXPERIMENT SUMMARY")
        print("="*60)
        
        data_info = results['data_collection']
        print(f"Data Collection: {data_info['total_samples']} total samples")
        print(f"  - LLM samples: {data_info['llm_samples']}")
        print(f"  - Human samples: {data_info['human_samples']}")
        print(f"  - API calls used: {data_info['api_calls_used']}/{MAX_API_CALLS}")
        print(f"  - Meets minimum requirement: {data_info['meets_minimum']}")
        
        print("\nDirectional Hypothesis Results:")
        for feature, analysis in results['feature_analysis'].items():
            prediction = analysis['directional_prediction']
            supported = "✓" if analysis['hypothesis_supported'] else "✗"
            effect_size = analysis['effect_size']['cohens_d']
            ci_lower = analysis['effect_size']['ci_lower'] 
            ci_upper = analysis['effect_size']['ci_upper']
            
            print(f"  {supported} {feature}: {prediction}")
            print(f"    Effect size (Cohen's d): {effect_size:.3f} [95% CI: {ci_lower:.3f}, {ci_upper:.3f}]")
            print(f"    Practical significance: {analysis['practical_significance']}")
        
        conclusions = results['conclusions']
        print(f"\nOverall Findings:")
        print(f"  - Hypotheses supported: {len(conclusions['hypotheses_supported'])}/3")
        print(f"  - Features with significant effects: {len(conclusions['significant_effects'])}")
        print(f"  - Overall hypothesis support: {conclusions['overall_support']}")
        
        for finding in conclusions['key_findings']:
            print(f"  - {finding}")
            
        print("\n" + "="*60)
        
    except Exception as e:
        print(f"Experiment failed with error: {e}")
        # Save error info
        error_results = {
            'error': str(e),
            'status': 'failed',
            'timestamp': time.time()
        }
        try:
            with open(RESULTS_PATH, 'w') as f:
                json.dump(error_results, f, indent=2)
        except:
            pass
        raise

if __name__ == "__main__":
    main()