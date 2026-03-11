import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from openai import OpenAI
import time
import traceback

def generate_math_problems(n_problems=15):
    """Generate math word problems with both direct and CoT solutions"""
    client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
    
    problems = []
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{
                "role": "user",
                "content": f"Generate {n_problems} diverse math word problems covering different topics (arithmetic, geometry, algebra, percentages, etc.). Each problem should be 1-2 sentences. Return as a numbered list."
            }],
            max_completion_tokens=800
        )
        
        problem_text = response.choices[0].message.content
        problems = [line.strip() for line in problem_text.split('\n') if line.strip() and any(char.isdigit() for char in line)]
        problems = [p.split('. ', 1)[-1] if '. ' in p else p for p in problems]
        problems = problems[:n_problems]
        
    except Exception as e:
        print(f"Error generating problems: {e}")
        # Fallback problems
        problems = [
            "A store sells apples for $2 per pound. How much do 3.5 pounds cost?",
            "What is the area of a rectangle with length 8 meters and width 5 meters?",
            "If 25% of students in a class of 40 wear glasses, how many students wear glasses?",
            "A car travels 180 miles in 3 hours. What is its average speed?",
            "What is 15% of 80?"
        ]
    
    return problems

def solve_problems(problems, use_cot=False):
    """Solve problems using either direct or chain-of-thought prompting"""
    client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
    solutions = []
    
    for i, problem in enumerate(problems):
        try:
            if use_cot:
                prompt = f"Solve this step by step, showing your reasoning:\n{problem}\n\nLet me think through this step by step:"
            else:
                prompt = f"Solve this problem:\n{problem}"
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=200
            )
            
            solution = response.choices[0].message.content
            solutions.append(solution)
            
            # Rate limiting
            if i < len(problems) - 1:
                time.sleep(0.5)
                
        except Exception as e:
            print(f"Error solving problem {i}: {e}")
            solutions.append("Unable to solve")
    
    return solutions

def compute_embeddings(texts, model_name='all-MiniLM-L6-v2'):
    """Compute embeddings for texts using sentence transformers"""
    try:
        model = SentenceTransformer(model_name)
        embeddings = model.encode(texts)
        return embeddings
    except Exception as e:
        print(f"Error computing embeddings: {e}")
        return None

def analyze_clustering(embeddings, labels, condition_name):
    """Analyze clustering quality using silhouette score"""
    if embeddings is None or len(embeddings) < 2:
        return {"silhouette_score": 0, "n_clusters": 0, "condition": condition_name}
    
    best_score = -1
    best_k = 2
    
    # Try different numbers of clusters
    for k in range(2, min(8, len(embeddings))):
        try:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings)
            score = silhouette_score(embeddings, cluster_labels)
            
            if score > best_score:
                best_score = score
                best_k = k
        except Exception as e:
            print(f"Clustering error for k={k}: {e}")
            continue
    
    return {
        "silhouette_score": best_score,
        "n_clusters": best_k,
        "condition": condition_name,
        "n_samples": len(embeddings)
    }

def create_visualization(direct_embeddings, cot_embeddings, save_path=None):
    """Create visualization of embedding distributions"""
    try:
        # Combine embeddings for PCA
        all_embeddings = np.vstack([direct_embeddings, cot_embeddings])
        labels = ['Direct'] * len(direct_embeddings) + ['CoT'] * len(cot_embeddings)
        
        # Apply PCA
        pca = PCA(n_components=2, random_state=42)
        embeddings_2d = pca.fit_transform(all_embeddings)
        
        # Create plot
        plt.figure(figsize=(10, 6))
        
        direct_2d = embeddings_2d[:len(direct_embeddings)]
        cot_2d = embeddings_2d[len(direct_embeddings):]
        
        plt.scatter(direct_2d[:, 0], direct_2d[:, 1], alpha=0.7, label='Direct Prompting', s=60)
        plt.scatter(cot_2d[:, 0], cot_2d[:, 1], alpha=0.7, label='Chain-of-Thought', s=60)
        
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.title('Embedding Space Distribution: Direct vs Chain-of-Thought')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path.replace('.json', '_embeddings.png'), dpi=150, bbox_inches='tight')
        
        plt.show()
        
    except Exception as e:
        print(f"Visualization error: {e}")

def main():
    """Main experiment function"""
    results = {
        "experiment_type": "embedding_study",
        "hypothesis": "Chain-of-thought prompting will show measurably different semantic clustering patterns",
        "timestamp": time.time(),
        "success": False,
        "error": None
    }
    
    try:
        print("Starting embedding clustering experiment...")
        print("=" * 60)
        
        # Generate problems
        print("1. Generating math word problems...")
        problems = generate_math_problems(15)
        print(f"Generated {len(problems)} problems")
        
        # Solve with both approaches
        print("\n2. Solving problems with direct prompting...")
        direct_solutions = solve_problems(problems, use_cot=False)
        
        print("\n3. Solving problems with chain-of-thought prompting...")
        cot_solutions = solve_problems(problems, use_cot=True)
        
        # Compute embeddings
        print("\n4. Computing embeddings...")
        direct_embeddings = compute_embeddings(direct_solutions)
        cot_embeddings = compute_embeddings(cot_solutions)
        
        if direct_embeddings is None or cot_embeddings is None:
            raise Exception("Failed to compute embeddings")
        
        # Analyze clustering
        print("\n5. Analyzing clustering patterns...")
        direct_clustering = analyze_clustering(direct_embeddings, None, "Direct")
        cot_clustering = analyze_clustering(cot_embeddings, None, "Chain-of-Thought")
        
        # Store results
        results.update({
            "n_problems": len(problems),
            "direct_clustering": direct_clustering,
            "cot_clustering": cot_clustering,
            "silhouette_difference": cot_clustering["silhouette_score"] - direct_clustering["silhouette_score"],
            "success_threshold": 0.05,
            "hypothesis_supported": cot_clustering["silhouette_score"] > direct_clustering["silhouette_score"] + 0.05,
            "success": True
        })
        
        # Create visualization
        print("\n6. Creating visualizations...")
        create_visualization(direct_embeddings, cot_embeddings, "/Users/kumacmini/cost-aware-research-search/results/iter_02_results.json")
        
        # Print results table
        print("\n" + "=" * 60)
        print("EXPERIMENT RESULTS")
        print("=" * 60)
        print(f"{'Condition':<20} {'Silhouette Score':<15} {'N Clusters':<12} {'N Samples':<10}")
        print("-" * 60)
        print(f"{'Direct':<20} {direct_clustering['silhouette_score']:<15.3f} {direct_clustering['n_clusters']:<12} {direct_clustering['n_samples']:<10}")
        print(f"{'Chain-of-Thought':<20} {cot_clustering['silhouette_score']:<15.3f} {cot_clustering['n_clusters']:<12} {cot_clustering['n_samples']:<10}")
        print("-" * 60)
        print(f"{'Difference':<20} {results['silhouette_difference']:<15.3f}")
        print(f"{'Success Threshold':<20} {results['success_threshold']:<15.3f}")
        print(f"{'Hypothesis Supported':<20} {str(results['hypothesis_supported']):<15}")
        print("=" * 60)
        
        if results['hypothesis_supported']:
            print("✓ HYPOTHESIS SUPPORTED: CoT shows better clustering than direct prompting")
        else:
            print("✗ HYPOTHESIS NOT SUPPORTED: No significant clustering improvement with CoT")
            
    except Exception as e:
        print(f"\nERROR: {e}")
        traceback.print_exc()
        results.update({
            "success": False,
            "error": str(e)
        })
    
    # Save results
    save_path = "/Users/kumacmini/cost-aware-research-search/results/iter_02_results.json"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {save_path}")

if __name__ == "__main__":
    main()