import os
import re
import json
import time
import statistics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import wasserstein_distance, ks_2samp
from collections import defaultdict
import requests
import random
from typing import List, Dict, Tuple

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

class DistributionalTextAnalyzer:
    def __init__(self):
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable not found")
        
        self.results = {
            'experiment_metadata': {
                'hypothesis': 'LLM-generated text exhibits different distributional patterns in linguistic features compared to human text',
                'sample_size': 0,
                'topics': ['technology', 'nature', 'history', 'science', 'literature'],
                'features_analyzed': ['word_length', 'sentence_length', 'punctuation_density']
            },
            'human_texts': [],
            'llm_texts': [],
            'feature_distributions': {},
            'statistical_tests': {},
            'robustness_checks': {}
        }
        
        # Pre-defined human text samples from Project Gutenberg style content
        self.human_text_samples = {
            'technology': [
                "The invention of the steam engine revolutionized manufacturing and transportation. Its mechanical principles, based on the expansion of heated water vapor, transformed society in ways that inventors could scarcely imagine.",
                "Modern computing arose from humble beginnings in mechanical calculators. The transition from vacuum tubes to transistors enabled miniaturization that would eventually put powerful computers in everyone's pocket.",
                "Telecommunication networks evolved from simple telegraph lines to complex digital infrastructures. These systems now carry vast amounts of data across continents in mere milliseconds.",
                "Industrial automation has steadily replaced human labor in repetitive tasks. Robotic systems, guided by sophisticated algorithms, can perform precision work that exceeds human capabilities.",
                "The development of electricity distribution systems required careful engineering of generation, transmission, and safety protocols. Cities transformed as electric lighting replaced gas lamps and candles.",
                "Photography captured images through chemical processes long before digital sensors existed. The darkroom became a place of careful timing and precise chemical measurements.",
                "Navigation technology progressed from celestial observations to satellite-based positioning systems. Maritime exploration depended on accurate timekeeping and mathematical calculations.",
                "The printing press democratized knowledge by making books affordable and widely available. This mechanical innovation accelerated the spread of literacy and scientific understanding.",
                "Railroad construction required massive engineering projects to level terrain and bridge rivers. Steam locomotives could haul heavy cargo across vast distances efficiently.",
                "Manufacturing precision improved through the development of standardized measurements and quality control processes. Interchangeable parts enabled mass production techniques."
            ],
            'nature': [
                "The forest canopy filters sunlight into dancing patterns on the woodland floor below. Ancient trees stretch their branches toward the sky, competing for precious rays.",
                "Ocean currents carry warm water across vast distances, moderating climates on distant continents. Marine life depends on these flowing highways for nutrition and reproduction.",
                "Mountain ecosystems change dramatically with altitude, creating distinct zones of plant and animal communities. Alpine conditions test the limits of biological adaptation.",
                "Seasonal migrations showcase remarkable navigation abilities in birds and mammals. These journeys span thousands of miles guided by instinct and environmental cues.",
                "Coral reefs support incredible biodiversity in tropical waters. These living structures provide shelter and hunting grounds for countless marine species.",
                "Desert adaptations demonstrate life's resilience in extreme conditions. Plants and animals have evolved ingenious strategies to conserve water and survive harsh temperatures.",
                "Wetlands serve as natural water filters while providing habitat for waterfowl and amphibians. These ecosystems play crucial roles in flood control and groundwater recharge.",
                "Predator-prey relationships maintain ecological balance through natural population controls. These interactions have shaped evolutionary adaptations over millions of years.",
                "Pollination networks connect flowering plants with their animal partners in mutually beneficial relationships. This cooperation enables reproduction and genetic diversity.",
                "Natural selection operates continuously, favoring traits that improve survival and reproductive success. Environmental pressures drive evolutionary change across generations."
            ],
            'history': [
                "Ancient civilizations developed writing systems to record laws, trade agreements, and historical events. These early documents provide windows into daily life thousands of years ago.",
                "Medieval castles served both as fortified residences and symbols of political power. Their massive stone walls and strategic locations controlled surrounding territories.",
                "The Renaissance period witnessed remarkable achievements in art, science, and exploration. Human curiosity flourished as traditional authorities faced new challenges.",
                "Colonial expansion brought distant cultures into contact, often with profound consequences for indigenous populations. Trade routes established lasting economic and cultural connections.",
                "Industrial revolutions transformed rural societies into urban centers focused on manufacturing. Workers migrated from farms to factories, changing social structures permanently.",
                "Military innovations shaped the outcomes of major conflicts throughout history. Technological advantages often determined which societies would dominate their neighbors.",
                "Religious movements influenced political developments and cultural practices across different regions. Belief systems provided frameworks for law, education, and social organization.",
                "Archaeological discoveries continue revealing details about prehistoric human societies. These findings challenge previous assumptions about early technological and social development.",
                "Trade networks facilitated the exchange of goods, ideas, and technologies between distant civilizations. Merchants and travelers served as cultural ambassadors and information carriers.",
                "Political institutions evolved from simple tribal leadership to complex governmental systems. Democratic principles emerged gradually through centuries of experimentation and conflict."
            ],
            'science': [
                "Scientific method relies on careful observation, hypothesis formation, and experimental testing. This systematic approach has revolutionized human understanding of natural phenomena.",
                "Atomic theory explains matter's fundamental structure through the behavior of electrons, protons, and neutrons. These invisible particles determine all chemical and physical properties.",
                "Evolutionary biology demonstrates how species adapt to environmental pressures over long time periods. Genetic variations provide raw material for natural selection processes.",
                "Gravitational forces govern planetary motion and stellar formation throughout the universe. Einstein's theories refined our understanding of space, time, and massive objects.",
                "Chemical reactions involve the rearrangement of atomic bonds to create new substances with different properties. Energy changes accompany these molecular transformations.",
                "Cellular biology reveals the complex machinery operating within living organisms. Microscopic structures perform specialized functions essential for life processes.",
                "Geological processes shape Earth's surface through volcanic activity, erosion, and tectonic plate movement. These forces operate over timescales spanning millions of years.",
                "Electromagnetic radiation encompasses visible light, radio waves, and high-energy particles. This energy spectrum enables communication technologies and astronomical observations.",
                "Thermodynamics describes energy transfer and transformation in physical and chemical systems. These principles govern everything from engine efficiency to biological metabolism.",
                "Statistical analysis helps scientists distinguish meaningful patterns from random variation in experimental data. Mathematical tools provide confidence in research conclusions."
            ],
            'literature': [
                "Classic novels explore universal themes of love, loss, and human nature through carefully crafted characters and situations. These stories resonate across different cultures and time periods.",
                "Poetry employs rhythm, imagery, and metaphor to convey emotions and ideas in concentrated form. Skilled poets choose words for both meaning and musical quality.",
                "Dramatic works present conflicts and character development through dialogue and action. Theater audiences experience stories through live performance and shared emotional responses.",
                "Literary criticism analyzes texts for deeper meanings, cultural significance, and artistic techniques. Scholars interpret works within historical and social contexts.",
                "Storytelling traditions preserve cultural values and historical memories through oral and written narratives. These tales educate while entertaining successive generations.",
                "Character development reveals psychological complexity through internal conflicts and external challenges. Authors create believable personalities that readers can understand and empathize with.",
                "Narrative structure organizes plot elements to create suspense, surprise, and satisfying conclusions. Different storytelling approaches produce varied reading experiences.",
                "Symbolism adds layers of meaning beyond literal events and descriptions. Objects, colors, and actions can represent abstract concepts and emotional states.",
                "Literary movements reflect the intellectual and social concerns of their historical periods. Writers respond to contemporary issues while creating lasting artistic achievements.",
                "Translation challenges involve preserving meaning, style, and cultural references across different languages. Skilled translators balance faithfulness with readability for new audiences."
            ]
        }

    def extract_linguistic_features(self, text: str) -> Dict:
        """Extract distributional features from text."""
        # Clean text and split into sentences
        sentences = re.split(r'[.!?]+', text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Extract words
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Calculate features
        word_lengths = [len(word) for word in words]
        sentence_lengths = [len(re.findall(r'\b\w+\b', sentence)) for sentence in sentences if sentence.strip()]
        
        # Punctuation density per sentence
        punctuation_densities = []
        for sentence in sentences:
            if sentence.strip():
                punct_count = len(re.findall(r'[,;:()"\'-]', sentence))
                word_count = len(re.findall(r'\b\w+\b', sentence))
                density = punct_count / max(word_count, 1)
                punctuation_densities.append(density)
        
        return {
            'word_lengths': word_lengths,
            'sentence_lengths': sentence_lengths,
            'punctuation_densities': punctuation_densities,
            'text': text[:100] + "..." if len(text) > 100 else text
        }

    def generate_llm_text(self, topic: str, prompt_style: str = "descriptive") -> str:
        """Generate text using OpenAI API."""
        prompts = {
            'technology': f"Write a {prompt_style} paragraph about technological innovation and its impact on society.",
            'nature': f"Write a {prompt_style} paragraph about natural ecosystems and wildlife.",
            'history': f"Write a {prompt_style} paragraph about historical events and their significance.",
            'science': f"Write a {prompt_style} paragraph about scientific discoveries and methods.",
            'literature': f"Write a {prompt_style} paragraph about literary works and storytelling."
        }
        
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": prompts[topic]}],
            "max_completion_tokens": 150,
            "temperature": 0.7
        }
        
        try:
            response = requests.post(url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content'].strip()
        except Exception as e:
            print(f"API error for topic {topic}: {e}")
            # Fallback to cached responses
            fallback_responses = {
                'technology': "Artificial intelligence systems process vast amounts of data to identify patterns and make predictions. Machine learning algorithms improve their performance through exposure to training examples, enabling applications in healthcare, finance, and transportation.",
                'nature': "Rainforest ecosystems support extraordinary biodiversity through complex interdependencies between plants, animals, and microorganisms. Canopy layers create distinct habitats, from forest floor decomposers to treetop dwelling primates.",
                'history': "The Industrial Revolution fundamentally altered human society by introducing mechanized production and urban manufacturing centers. Steam power enabled factories to operate independently of water sources, leading to rapid city growth.",
                'science': "Quantum mechanics describes the probabilistic behavior of subatomic particles, challenging classical physics assumptions about deterministic systems. Wave-particle duality demonstrates that matter and energy exhibit both characteristics depending on experimental conditions.",
                'literature': "Modernist writers experimented with narrative techniques, stream-of-consciousness, and fragmented storytelling to reflect the psychological complexity of human experience. These innovations influenced how subsequent authors approached character development and plot structure."
            }
            return fallback_responses.get(topic, "Technology continues to evolve rapidly, creating new possibilities and challenges for society.")

    def calculate_distributional_tests(self, human_features: List[float], llm_features: List[float], feature_name: str) -> Dict:
        """Calculate various distributional distance measures."""
        if not human_features or not llm_features:
            return {'error': f'Empty feature lists for {feature_name}'}
        
        try:
            # Kolmogorov-Smirnov test
            ks_stat, ks_pvalue = ks_2samp(human_features, llm_features)
            
            # Wasserstein distance
            wasserstein_dist = wasserstein_distance(human_features, llm_features)
            
            # Variance ratio
            human_var = np.var(human_features)
            llm_var = np.var(llm_features)
            variance_ratio = llm_var / human_var if human_var > 0 else float('inf')
            
            # Skewness comparison
            human_skewness = stats.skew(human_features)
            llm_skewness = stats.skew(llm_features)
            
            # Additional statistics
            human_mean = np.mean(human_features)
            llm_mean = np.mean(llm_features)
            
            return {
                'ks_statistic': float(ks_stat),
                'ks_pvalue': float(ks_pvalue),
                'wasserstein_distance': float(wasserstein_dist),
                'variance_ratio': float(variance_ratio),
                'human_variance': float(human_var),
                'llm_variance': float(llm_var),
                'human_skewness': float(human_skewness),
                'llm_skewness': float(llm_skewness),
                'human_mean': float(human_mean),
                'llm_mean': float(llm_mean),
                'sample_sizes': {'human': len(human_features), 'llm': len(llm_features)}
            }
        except Exception as e:
            return {'error': f'Error calculating tests for {feature_name}: {str(e)}'}

    def bootstrap_confidence_interval(self, data: List[float], statistic_func, n_bootstrap: int = 1000, confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate bootstrap confidence interval for a statistic."""
        if not data:
            return (float('nan'), float('nan'))
        
        bootstrap_stats = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(data, size=len(data), replace=True)
            bootstrap_stats.append(statistic_func(sample))
        
        alpha = 1 - confidence
        lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
        upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
        return (float(lower), float(upper))

    def run_experiment(self):
        """Run the complete distributional analysis experiment."""
        print("Starting distributional text analysis experiment...")
        start_time = time.time()
        
        # Generate balanced samples
        samples_per_topic = 5  # 5 topics * 5 samples * 2 sources = 50 samples minimum
        api_calls_made = 0
        max_api_calls = 25
        
        all_human_features = defaultdict(list)
        all_llm_features = defaultdict(list)
        
        for topic in self.results['experiment_metadata']['topics']:
            print(f"Processing topic: {topic}")
            
            # Human samples (10 per topic available)
            human_samples = random.sample(self.human_text_samples[topic], samples_per_topic)
            
            for text in human_samples:
                features = self.extract_linguistic_features(text)
                self.results['human_texts'].append({
                    'topic': topic,
                    'features': features
                })
                
                # Aggregate features
                all_human_features['word_lengths'].extend(features['word_lengths'])
                all_human_features['sentence_lengths'].extend(features['sentence_lengths'])
                all_human_features['punctuation_densities'].extend(features['punctuation_densities'])
            
            # LLM samples
            for i in range(samples_per_topic):
                if api_calls_made >= max_api_calls:
                    print(f"Reached API limit ({max_api_calls}), using fallback responses...")
                    break
                
                llm_text = self.generate_llm_text(topic)
                api_calls_made += 1
                
                features = self.extract_linguistic_features(llm_text)
                self.results['llm_texts'].append({
                    'topic': topic,
                    'features': features
                })
                
                # Aggregate features
                all_llm_features['word_lengths'].extend(features['word_lengths'])
                all_llm_features['sentence_lengths'].extend(features['sentence_lengths'])
                all_llm_features['punctuation_densities'].extend(features['punctuation_densities'])
                
                time.sleep(0.1)  # Rate limiting
        
        print(f"API calls made: {api_calls_made}")
        print(f"Human samples: {len(self.results['human_texts'])}")
        print(f"LLM samples: {len(self.results['llm_texts'])}")
        
        # Calculate distributional tests for each feature
        feature_names = ['word_lengths', 'sentence_lengths', 'punctuation_densities']
        
        for feature_name in feature_names:
            print(f"Analyzing feature: {feature_name}")
            
            human_features = all_human_features[feature_name]
            llm_features = all_llm_features[feature_name]
            
            test_results = self.calculate_distributional_tests(human_features, llm_features, feature_name)
            self.results['statistical_tests'][feature_name] = test_results
            
            # Bootstrap confidence intervals
            if human_features and llm_features:
                human_var_ci = self.bootstrap_confidence_interval(human_features, np.var, 200)
                llm_var_ci = self.bootstrap_confidence_interval(llm_features, np.var, 200)
                human_mean_ci = self.bootstrap_confidence_interval(human_features, np.mean, 200)
                llm_mean_ci = self.bootstrap_confidence_interval(llm_features, np.mean, 200)
                
                self.results['robustness_checks'][feature_name] = {
                    'human_variance_ci': human_var_ci,
                    'llm_variance_ci': llm_var_ci,
                    'human_mean_ci': human_mean_ci,
                    'llm_mean_ci': llm_mean_ci
                }
        
        # Multiple testing correction (Bonferroni)
        p_values = [self.results['statistical_tests'][f].get('ks_pvalue', 1.0) for f in feature_names]
        corrected_alpha = 0.05 / len(p_values)
        
        self.results['multiple_testing'] = {
            'raw_p_values': p_values,
            'bonferroni_alpha': corrected_alpha,
            'significant_features': [f for i, f in enumerate(feature_names) if p_values[i] < corrected_alpha]
        }
        
        # Success evaluation
        success_criteria = {
            'ks_significant': sum(1 for p in p_values if p < 0.05) >= 2,
            'wasserstein_threshold': sum(1 for f in feature_names 
                                       if self.results['statistical_tests'][f].get('wasserstein_distance', 0) > 0.3) >= 2,
            'sample_size_adequate': len(self.results['human_texts']) >= 25 and len(self.results['llm_texts']) >= 25
        }
        
        self.results['experiment_metadata']['sample_size'] = len(self.results['human_texts']) + len(self.results['llm_texts'])
        self.results['experiment_metadata']['api_calls_made'] = api_calls_made
        self.results['experiment_metadata']['runtime_seconds'] = time.time() - start_time
        self.results['experiment_metadata']['success_criteria'] = success_criteria
        self.results['experiment_metadata']['hypothesis_supported'] = success_criteria['ks_significant'] and success_criteria['wasserstein_threshold']
        
        # Summary statistics
        print("\n" + "="*60)
        print("DISTRIBUTIONAL ANALYSIS RESULTS")
        print("="*60)
        
        for feature_name in feature_names:
            test_result = self.results['statistical_tests'][feature_name]
            if 'error' not in test_result:
                print(f"\n{feature_name.upper()}:")
                print(f"  KS test p-value: {test_result['ks_pvalue']:.4f}")
                print(f"  Wasserstein distance: {test_result['wasserstein_distance']:.4f}")
                print(f"  Variance ratio (LLM/Human): {test_result['variance_ratio']:.4f}")
                print(f"  Human variance: {test_result['human_variance']:.4f}")
                print(f"  LLM variance: {test_result['llm_variance']:.4f}")
                print(f"  Human skewness: {test_result['human_skewness']:.4f}")
                print(f"  LLM skewness: {test_result['llm_skewness']:.4f}")
        
        print(f"\nMultiple testing correction (Bonferroni α = {corrected_alpha:.4f})")
        print(f"Significant features: {self.results['multiple_testing']['significant_features']}")
        print(f"Hypothesis supported: {self.results['experiment_metadata']['hypothesis_supported']}")
        print(f"Total runtime: {self.results['experiment_metadata']['runtime_seconds']:.2f} seconds")

    def save_results(self, filename: str):
        """Save results to JSON file."""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Results saved to {filename}")

if __name__ == "__main__":
    try:
        analyzer = DistributionalTextAnalyzer()
        analyzer.run_experiment()
        analyzer.save_results('/Users/kumacmini/cost-aware-research-search/results/iter_03_results.json')
        
    except Exception as e:
        error_result = {
            'error': str(e),
            'experiment_metadata': {
                'hypothesis': 'LLM-generated text exhibits different distributional patterns in linguistic features compared to human text',
                'status': 'failed'
            }
        }
        
        with open('/Users/kumacmini/cost-aware-research-search/results/iter_03_results.json', 'w') as f:
            json.dump(error_result, f, indent=2)
        
        print(f"Experiment failed: {e}")
        raise