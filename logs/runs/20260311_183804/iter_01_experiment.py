import os
import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def generate_math_problems():
    """Generate a set of math problems for evaluation."""
    problems = [
        "What is 15 × 23?",
        "Solve for x: 2x + 7 = 19",
        "Find the area of a circle with radius 5",
        "What is the square root of 144?",
        "Calculate 7! (7 factorial)",
        "If a train travels 60 mph for 2.5 hours, how far does it go?",
        "What is 25% of 80?",
        "Solve: 3x² - 12x + 9 = 0",
        "Find the slope of the line passing through (2,3) and (5,9)",
        "What is the sum of angles in a triangle?",
        "Calculate the volume of a cube with side length 4",
        "What is log₁₀(1000)?",
        "Find the derivative of x³ + 2x²",
        "What is sin(30°)?",
        "Calculate 2⁸",
        "Find the median of: 3, 7, 2, 9, 5, 1, 8",
        "What is the perimeter of a rectangle 6×4?",
        "Solve: |x - 3| = 5",
        "Calculate the compound interest on $1000 at 5% for 2 years",
        "What is the greatest common divisor of 24 and 36?",
        "Find the distance between points (1,2) and (4,6)",
        "What is 0.25 as a fraction?",
        "Calculate the surface area of a sphere with radius 3",
        "Solve for y: 3y - 5 = 2y + 8",
        "What is the probability of rolling a 6 on a fair die?",
        "Find the nth term formula for: 2, 5, 8, 11, ...",
        "Calculate √(49 + 25)",
        "What is 3/4 + 2/3?",
        "Find the x-intercept of y = 2x - 6",
        "What is the circumference of a circle with diameter 10?",
    ]
    return problems[:25]  # Use 25 problems to stay within API limits

def create_direct_prompts(problems):
    """Create direct prompts for math problems."""
    return [f"Solve this problem: {problem}" for problem in problems]

def create_cot_prompts(problems):
    """Create chain-of-thought prompts for math problems."""
    cot_prompts = []
    for problem in problems:
        prompt = f"Solve this problem step by step, showing your reasoning: {problem}\n\nLet me think through this step by step:\n1. First, I need to identify what type of problem this is\n2. Then I'll apply the appropriate method\n3. Finally, I'll check my answer"
        cot_prompts.append(prompt)
    return cot_prompts

def calculate_embedding_metrics(embeddings, label):
    """Calculate various metrics for embeddings."""
    # Calculate pairwise cosine similarities
    similarities = cosine_similarity(embeddings)
    
    # Remove diagonal (self-similarities)
    mask = np.eye(similarities.shape[0], dtype=bool)
    similarities_no_diag = similarities[~mask]
    
    # Calculate variance in similarities (our main metric)
    similarity_variance = np.var(similarities_no_diag)
    
    # Calculate mean similarity
    mean_similarity = np.mean(similarities_no_diag)
    
    # Calculate cluster tightness using K-means
    try:
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(embeddings)
        inertia = kmeans.inertia_
    except:
        inertia = None
    
    return {
        'similarity_variance': float(similarity_variance),
        'mean_similarity': float(mean_similarity),
        'cluster_inertia': float(inertia) if inertia is not None else None,
        'n_samples': len(embeddings)
    }

def main():
    results = {
        'experiment': 'embedding_study',
        'hypothesis': 'CoT prompting shows different embedding patterns with tighter clustering',
        'timestamp': pd.Timestamp.now().isoformat(),
        'error': None,
        'success': False
    }
    
    try:
        print("Starting embedding analysis experiment...")
        print("=" * 60)
        
        # Generate math problems
        problems = generate_math_problems()
        print(f"Generated {len(problems)} math problems")
        
        # Create prompts
        direct_prompts = create_direct_prompts(problems)
        cot_prompts = create_cot_prompts(problems)
        
        print("Loading sentence transformer model...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Generate embeddings
        print("Generating embeddings for direct prompts...")
        direct_embeddings = model.encode(direct_prompts)
        
        print("Generating embeddings for CoT prompts...")
        cot_embeddings = model.encode(cot_prompts)
        
        print(f"Direct embeddings shape: {direct_embeddings.shape}")
        print(f"CoT embeddings shape: {cot_embeddings.shape}")
        
        # Calculate metrics
        print("\nCalculating metrics...")
        direct_metrics = calculate_embedding_metrics(direct_embeddings, "direct")
        cot_metrics = calculate_embedding_metrics(cot_embeddings, "cot")
        
        # Calculate variance reduction
        variance_reduction = ((direct_metrics['similarity_variance'] - cot_metrics['similarity_variance']) / 
                            direct_metrics['similarity_variance']) * 100
        
        # Determine success
        success_threshold = 20.0  # 20% reduction
        success = variance_reduction >= success_threshold
        
        results.update({
            'direct_metrics': direct_metrics,
            'cot_metrics': cot_metrics,
            'variance_reduction_percent': float(variance_reduction),
            'success_threshold': success_threshold,
            'success': success,
            'n_problems': len(problems)
        })
        
        # Create results table
        print("\n" + "=" * 60)
        print("EXPERIMENT RESULTS")
        print("=" * 60)
        
        results_df = pd.DataFrame({
            'Metric': [
                'Similarity Variance',
                'Mean Similarity',
                'Cluster Inertia',
                'Sample Count'
            ],
            'Direct Prompts': [
                f"{direct_metrics['similarity_variance']:.6f}",
                f"{direct_metrics['mean_similarity']:.4f}",
                f"{direct_metrics['cluster_inertia']:.2f}" if direct_metrics['cluster_inertia'] else "N/A",
                f"{direct_metrics['n_samples']}"
            ],
            'CoT Prompts': [
                f"{cot_metrics['similarity_variance']:.6f}",
                f"{cot_metrics['mean_similarity']:.4f}",
                f"{cot_metrics['cluster_inertia']:.2f}" if cot_metrics['cluster_inertia'] else "N/A",
                f"{cot_metrics['n_samples']}"
            ]
        })
        
        print(results_df.to_string(index=False))
        print()
        
        print(f"Variance Reduction: {variance_reduction:.2f}%")
        print(f"Success Threshold: {success_threshold}%")
        print(f"Experiment Success: {'YES' if success else 'NO'}")
        
        if success:
            print("✅ Hypothesis supported: CoT prompts show tighter embedding clustering")
        else:
            print("❌ Hypothesis not supported: No significant clustering difference found")
        
        print("\nInterpretation:")
        if variance_reduction > 0:
            print(f"- CoT embeddings show {variance_reduction:.1f}% less variance in similarity")
            print("- This suggests more consistent semantic patterns in CoT prompts")
        else:
            print(f"- CoT embeddings show {abs(variance_reduction):.1f}% more variance")
            print("- Direct prompts may be more semantically consistent")
        
    except Exception as e:
        print(f"Experiment failed with error: {str(e)}")
        results['error'] = str(e)
        results['success'] = False
    
    finally:
        # Save results
        output_path = "/Users/kumacmini/cost-aware-research-search/results/iter_01_results.json"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")
        print("Experiment completed!")

if __name__ == "__main__":
    main()