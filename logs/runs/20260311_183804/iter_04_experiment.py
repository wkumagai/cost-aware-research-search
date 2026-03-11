import os
import json
import numpy as np
import requests
import time
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def get_wikipedia_samples(n_samples=100):
    """Get general domain text samples from Wikipedia API"""
    samples = []
    try:
        # Get random article titles
        url = "https://en.wikipedia.org/api/rest_v1/page/random/summary"
        for i in range(min(n_samples, 20)):  # Limit API calls
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    extract = data.get('extract', '')
                    if len(extract) > 50:  # Ensure meaningful content
                        samples.append(extract)
                time.sleep(0.1)  # Rate limiting
            except:
                continue
        
        # Fill remaining with hardcoded samples if needed
        hardcoded_general = [
            "The weather today is sunny with temperatures reaching 75 degrees Fahrenheit.",
            "Cooking pasta requires boiling water and adding salt for flavor enhancement.",
            "Many people enjoy reading books in their spare time for entertainment.",
            "Transportation systems include buses, trains, and automobiles for daily commuting.",
            "Music has the power to evoke emotions and create lasting memories.",
            "Exercise is beneficial for maintaining physical health and mental well-being.",
            "Technology continues to advance rapidly, changing how we communicate.",
            "Education plays a crucial role in personal and professional development.",
            "Art galleries showcase diverse works from various cultural backgrounds.",
            "Gardening provides relaxation while producing fresh vegetables and flowers."
        ]
        
        while len(samples) < n_samples and hardcoded_general:
            samples.append(hardcoded_general.pop(0))
            
    except Exception as e:
        print(f"Wikipedia API error: {e}")
        
    return samples[:n_samples]

def get_medical_samples(n_samples=100):
    """Get medical domain text samples"""
    hardcoded_medical = [
        "Myocardial infarction occurs when coronary artery occlusion leads to cardiomyocyte necrosis and subsequent ventricular dysfunction.",
        "Pharmacokinetic properties of beta-lactam antibiotics demonstrate time-dependent bactericidal activity against gram-positive pathogens.",
        "Hepatic cytochrome P450 enzymes metabolize xenobiotics through oxidative biotransformation pathways in hepatocytes.",
        "Glomerular filtration rate decreases with progressive nephrosclerosis and tubular atrophy in chronic kidney disease.",
        "Immunoglobulin G antibodies provide long-term humoral immunity through complement activation and opsonization mechanisms.",
        "Adenocarcinoma of the pancreas exhibits high metastatic potential due to early lymphovascular invasion.",
        "Dopaminergic neurons in the substantia nigra undergo progressive degeneration in Parkinson's disease pathophysiology.",
        "Thromboembolism risk increases with prolonged immobilization due to venous stasis and hypercoagulability.",
        "Bronchial hyperresponsiveness in asthma results from smooth muscle contraction and inflammatory mediator release.",
        "Insulin resistance develops through impaired glucose uptake in peripheral tissues and hepatic gluconeogenesis.",
        "Atherosclerotic plaque formation involves endothelial dysfunction, lipid accumulation, and macrophage infiltration.",
        "Antimicrobial resistance emerges through selective pressure and horizontal gene transfer mechanisms.",
        "Osteoporosis develops when bone resorption exceeds formation, leading to decreased bone mineral density.",
        "Cardiac arrhythmias result from abnormal electrical conduction through specialized myocardial tissue.",
        "Malignant transformation involves oncogene activation and tumor suppressor gene inactivation pathways."
    ]
    return hardcoded_medical[:n_samples]

def get_legal_samples(n_samples=100):
    """Get legal domain text samples"""
    hardcoded_legal = [
        "The plaintiff's motion for summary judgment lacks sufficient evidence to establish prima facie liability under tort law.",
        "Contractual obligations require consideration, mutual assent, and legal capacity to form binding agreements.",
        "Statutory construction principles dictate that legislative intent governs judicial interpretation of ambiguous provisions.",
        "Due process violations occur when defendants are denied fundamental procedural safeguards during adjudication.",
        "Intellectual property infringement requires proof of ownership, validity, and unauthorized use of protected works.",
        "Corporate fiduciary duties encompass loyalty and care obligations owed by directors to shareholders.",
        "Criminal liability requires mens rea and actus reus elements to establish guilt beyond reasonable doubt.",
        "Constitutional jurisprudence establishes precedential authority through stare decisis doctrine application.",
        "Administrative law governs agency rulemaking procedures and judicial review standards for regulatory actions.",
        "Property rights encompass fee simple absolute ownership and various forms of concurrent estate interests.",
        "Evidentiary rules determine admissibility of testimonial and documentary proof at trial proceedings.",
        "Appellate jurisdiction allows higher courts to review lower tribunal decisions for legal error.",
        "Injunctive relief provides equitable remedies when monetary damages prove inadequate compensation.",
        "Securities regulations mandate disclosure requirements for publicly traded corporations and investment advisors.",
        "Bankruptcy proceedings involve automatic stays, creditor priorities, and debtor discharge provisions."
    ]
    return hardcoded_legal[:n_samples]

def compute_embeddings(texts, model_name='all-MiniLM-L6-v2'):
    """Compute sentence embeddings for given texts"""
    try:
        model = SentenceTransformer(model_name)
        embeddings = model.encode(texts, show_progress_bar=True)
        return embeddings
    except Exception as e:
        print(f"Embedding computation error: {e}")
        return None

def evaluate_clustering_quality(embeddings, n_clusters=3):
    """Evaluate clustering quality using silhouette score"""
    if embeddings is None or len(embeddings) < n_clusters:
        return 0.0
    
    try:
        # Standardize embeddings
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings)
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings_scaled)
        
        # Calculate silhouette score
        if len(set(cluster_labels)) > 1:
            score = silhouette_score(embeddings_scaled, cluster_labels)
            return score
        else:
            return 0.0
    except Exception as e:
        print(f"Clustering evaluation error: {e}")
        return 0.0

def main():
    print("Starting Semantic Clustering Quality Experiment")
    print("=" * 50)
    
    results = {
        "experiment": "semantic_clustering_domain_comparison",
        "hypothesis": "Semantic clustering quality degrades with domain-specific jargon",
        "conditions": {},
        "summary": {},
        "success": False
    }
    
    # Sample sizes per domain
    samples_per_domain = 100
    
    try:
        # Collect text samples from different domains
        print("Collecting text samples...")
        
        print("- Fetching general domain texts (Wikipedia)...")
        general_texts = get_wikipedia_samples(samples_per_domain)
        print(f"  Collected {len(general_texts)} general texts")
        
        print("- Collecting medical domain texts...")
        medical_texts = get_medical_samples(samples_per_domain)
        print(f"  Collected {len(medical_texts)} medical texts")
        
        print("- Collecting legal domain texts...")
        legal_texts = get_legal_samples(samples_per_domain)
        print(f"  Collected {len(legal_texts)} legal texts")
        
        # Compute embeddings for each domain
        print("\nComputing sentence embeddings...")
        
        print("- Computing general domain embeddings...")
        general_embeddings = compute_embeddings(general_texts)
        
        print("- Computing medical domain embeddings...")
        medical_embeddings = compute_embeddings(medical_texts)
        
        print("- Computing legal domain embeddings...")
        legal_embeddings = compute_embeddings(legal_texts)
        
        # Evaluate clustering quality
        print("\nEvaluating clustering quality...")
        
        general_silhouette = evaluate_clustering_quality(general_embeddings)
        medical_silhouette = evaluate_clustering_quality(medical_embeddings)
        legal_silhouette = evaluate_clustering_quality(legal_embeddings)
        
        print(f"General domain silhouette score: {general_silhouette:.4f}")
        print(f"Medical domain silhouette score: {medical_silhouette:.4f}")
        print(f"Legal domain silhouette score: {legal_silhouette:.4f}")
        
        # Calculate percentage differences
        medical_diff = ((general_silhouette - medical_silhouette) / general_silhouette) * 100 if general_silhouette > 0 else 0
        legal_diff = ((general_silhouette - legal_silhouette) / general_silhouette) * 100 if general_silhouette > 0 else 0
        
        # Store results
        results["conditions"] = {
            "general": {
                "domain": "general",
                "n_samples": len(general_texts),
                "silhouette_score": float(general_silhouette),
                "texts_sample": general_texts[:3]
            },
            "medical": {
                "domain": "medical",
                "n_samples": len(medical_texts),
                "silhouette_score": float(medical_silhouette),
                "difference_percent": float(medical_diff),
                "texts_sample": medical_texts[:3]
            },
            "legal": {
                "domain": "legal",
                "n_samples": len(legal_texts),
                "silhouette_score": float(legal_silhouette),
                "difference_percent": float(legal_diff),
                "texts_sample": legal_texts[:3]
            }
        }
        
        # Summary statistics
        avg_domain_specific_degradation = (medical_diff + legal_diff) / 2
        hypothesis_supported = avg_domain_specific_degradation > 15.0
        
        results["summary"] = {
            "baseline_silhouette": float(general_silhouette),
            "medical_degradation_percent": float(medical_diff),
            "legal_degradation_percent": float(legal_diff),
            "average_degradation_percent": float(avg_domain_specific_degradation),
            "hypothesis_supported": hypothesis_supported,
            "success_threshold_met": avg_domain_specific_degradation > 15.0
        }
        
        results["success"] = hypothesis_supported
        
        # Print results table
        print("\n" + "=" * 70)
        print("SEMANTIC CLUSTERING QUALITY RESULTS")
        print("=" * 70)
        print(f"{'Domain':<15} {'Silhouette':<12} {'Degradation':<12} {'N Samples':<10}")
        print("-" * 70)
        print(f"{'General':<15} {general_silhouette:<12.4f} {'-':<12} {len(general_texts):<10}")
        print(f"{'Medical':<15} {medical_silhouette:<12.4f} {medical_diff:<12.1f}% {len(medical_texts):<10}")
        print(f"{'Legal':<15} {legal_silhouette:<12.4f} {legal_diff:<12.1f}% {len(legal_texts):<10}")
        print("-" * 70)
        print(f"Average domain-specific degradation: {avg_domain_specific_degradation:.1f}%")
        print(f"Hypothesis supported: {'YES' if hypothesis_supported else 'NO'}")
        print(f"Success threshold (>15% degradation): {'MET' if avg_domain_specific_degradation > 15.0 else 'NOT MET'}")
        
    except Exception as e:
        print(f"Experiment error: {e}")
        results["error"] = str(e)
    
    # Save results
    output_path = "/Users/kumacmini/cost-aware-research-search/results/iter_04_results.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    print("Experiment completed!")

if __name__ == "__main__":
    main()