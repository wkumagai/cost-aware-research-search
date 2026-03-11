import json
import os
import random
import time
import statistics
from collections import defaultdict
import openai

# Initialize OpenAI client
client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def create_logical_reasoning_problem():
    """Create a synthetic logical reasoning problem with multiple facts."""
    
    # Define problem templates with 4 supporting facts
    templates = [
        {
            'conclusion': 'X is the best candidate for the position',
            'facts': [
                'X has 10 years of relevant experience',
                'X scored highest on the technical assessment', 
                'X has excellent references from previous employers',
                'X has the required certifications for the role'
            ]
        },
        {
            'conclusion': 'The project will be completed on time',
            'facts': [
                'The development team has met all previous deadlines',
                'All necessary resources have been allocated',
                'The project scope is clearly defined',
                'No major technical blockers have been identified'
            ]
        },
        {
            'conclusion': 'The new policy will reduce costs',
            'facts': [
                'Similar policies reduced costs by 15% in other departments',
                'The policy eliminates redundant processes',
                'Implementation requires minimal additional resources',
                'Expert analysis predicts significant savings'
            ]
        }
    ]
    
    template = random.choice(templates)
    
    # Randomize specific details
    names = ['Alex', 'Jordan', 'Taylor', 'Casey', 'Morgan']
    companies = ['TechCorp', 'InnovateCo', 'DataSys', 'CloudTech', 'AI Solutions']
    
    conclusion = template['conclusion'].replace('X', random.choice(names))
    facts = [fact.replace('X', random.choice(names)) for fact in template['facts']]
    
    return conclusion, facts

def create_prompt(conclusion, facts, target_fact_position):
    """Create a prompt with facts in specified order, targeting one fact for importance."""
    
    # Create three orderings: target fact first, middle, last
    if target_fact_position == 'first':
        ordered_facts = facts
    elif target_fact_position == 'middle':
        # Put target fact (first fact) in middle positions
        ordered_facts = [facts[1], facts[0], facts[2], facts[3]]
    else:  # 'last'
        # Put target fact (first fact) at the end
        ordered_facts = facts[1:] + [facts[0]]
    
    fact_text = '\n'.join([f"{i+1}. {fact}" for i, fact in enumerate(ordered_facts)])
    
    prompt = f"""Based on the following facts, evaluate the conclusion and explain which fact is most important for your reasoning:

Facts:
{fact_text}

Conclusion: {conclusion}

Please:
1. Rate how strongly the facts support the conclusion (scale 1-10)
2. Identify which single fact is most important for reaching this conclusion
3. Explain your reasoning briefly

Format your response as:
Rating: [number]
Most important fact: [number 1-4]
Reasoning: [brief explanation]"""
    
    return prompt, ordered_facts

def query_llm(prompt):
    """Query the LLM with error handling and retries."""
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=150,
                temperature=0.1
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"API call failed (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                return None

def parse_response(response_text):
    """Parse the LLM response to extract rating and most important fact."""
    if not response_text:
        return None, None
    
    lines = response_text.split('\n')
    rating = None
    most_important_fact = None
    
    for line in lines:
        line = line.strip()
        if line.lower().startswith('rating:'):
            try:
                rating = int(line.split(':')[1].strip())
            except:
                pass
        elif line.lower().startswith('most important fact:'):
            try:
                most_important_fact = int(line.split(':')[1].strip())
            except:
                pass
    
    return rating, most_important_fact

def run_experiment():
    """Run the complete experiment."""
    print("Starting Information Ordering Effects Experiment")
    print("=" * 50)
    
    results = []
    api_call_count = 0
    max_api_calls = 30
    
    # Target 60 samples across 3 conditions (20 each)
    samples_per_condition = 20
    conditions = ['first', 'middle', 'last']
    
    condition_results = defaultdict(list)
    
    for condition in conditions:
        print(f"\nTesting condition: target fact in {condition} position")
        
        for sample_idx in range(samples_per_condition):
            if api_call_count >= max_api_calls:
                print(f"Reached API call limit ({max_api_calls})")
                break
                
            print(f"  Sample {sample_idx + 1}/{samples_per_condition}")
            
            # Create problem
            conclusion, facts = create_logical_reasoning_problem()
            
            # Create prompt with target fact in specified position
            prompt, ordered_facts = create_prompt(conclusion, facts, condition)
            
            # Query LLM
            response = query_llm(prompt)
            api_call_count += 1
            
            if response:
                rating, most_important_fact = parse_response(response)
                
                if rating is not None and most_important_fact is not None:
                    # Check if LLM selected the target fact (originally first fact)
                    if condition == 'first':
                        target_selected = (most_important_fact == 1)
                        target_position = 1
                    elif condition == 'middle':
                        target_selected = (most_important_fact == 2)
                        target_position = 2
                    else:  # 'last'
                        target_selected = (most_important_fact == 4)
                        target_position = 4
                    
                    result = {
                        'condition': condition,
                        'sample_id': sample_idx,
                        'conclusion': conclusion,
                        'facts': facts,
                        'ordered_facts': ordered_facts,
                        'rating': rating,
                        'most_important_fact': most_important_fact,
                        'target_selected': target_selected,
                        'target_position': target_position,
                        'response': response
                    }
                    
                    results.append(result)
                    condition_results[condition].append(target_selected)
            
            time.sleep(0.1)  # Rate limiting
        
        if api_call_count >= max_api_calls:
            break
    
    return results, condition_results, api_call_count

def analyze_results(results, condition_results):
    """Analyze the experimental results."""
    print("\n" + "="*50)
    print("ANALYSIS RESULTS")
    print("="*50)
    
    analysis = {}
    
    # Calculate selection rates for each condition
    for condition in condition_results:
        selections = condition_results[condition]
        if selections:
            selection_rate = statistics.mean(selections)
            analysis[condition] = {
                'selection_rate': selection_rate,
                'n_samples': len(selections),
                'n_selected': sum(selections)
            }
    
    # Calculate ordering effect
    if 'first' in analysis and 'last' in analysis:
        first_rate = analysis['first']['selection_rate']
        last_rate = analysis['last']['selection_rate']
        ordering_effect = first_rate - last_rate
        analysis['ordering_effect'] = ordering_effect
        analysis['meets_threshold'] = abs(ordering_effect) > 0.15
    
    # Statistical summary
    all_rates = [analysis[cond]['selection_rate'] for cond in analysis if cond in ['first', 'middle', 'last']]
    if len(all_rates) >= 2:
        analysis['variance'] = statistics.variance(all_rates) if len(all_rates) > 1 else 0
        analysis['range'] = max(all_rates) - min(all_rates)
    
    return analysis

def print_results_table(analysis, total_api_calls):
    """Print formatted results table."""
    print(f"\nTotal API calls used: {total_api_calls}")
    print(f"Total samples collected: {sum(analysis[cond]['n_samples'] for cond in analysis if cond in ['first', 'middle', 'last'])}")
    
    print("\nSelection Rate by Condition:")
    print("-" * 40)
    print(f"{'Condition':<12} {'Rate':<8} {'N':<4} {'Selected'}")
    print("-" * 40)
    
    for condition in ['first', 'middle', 'last']:
        if condition in analysis:
            data = analysis[condition]
            print(f"{condition:<12} {data['selection_rate']:.3f}    {data['n_samples']:<4} {data['n_selected']}")
    
    print("-" * 40)
    
    if 'ordering_effect' in analysis:
        print(f"\nOrdering Effect (First - Last): {analysis['ordering_effect']:.3f}")
        print(f"Meets Success Threshold (>0.15): {analysis['meets_threshold']}")
        
        if 'range' in analysis:
            print(f"Range across conditions: {analysis['range']:.3f}")
            print(f"Variance: {analysis['variance']:.4f}")

def main():
    """Main execution function."""
    start_time = time.time()
    
    try:
        # Run experiment
        results, condition_results, api_calls = run_experiment()
        
        # Analyze results
        analysis = analyze_results(results, condition_results)
        
        # Print results
        print_results_table(analysis, api_calls)
        
        # Save results
        output_data = {
            'experiment_type': 'information_ordering_effects',
            'total_samples': len(results),
            'api_calls_used': api_calls,
            'execution_time_seconds': time.time() - start_time,
            'analysis': analysis,
            'raw_results': results,
            'success_criteria': {
                'min_samples': 50,
                'success_threshold': 0.15,
                'met_min_samples': len(results) >= 50,
                'met_threshold': analysis.get('meets_threshold', False)
            }
        }
        
        # Ensure results directory exists
        os.makedirs('/Users/kumacmini/cost-aware-research-search/results', exist_ok=True)
        
        # Save to JSON
        output_path = '/Users/kumacmini/cost-aware-research-search/results/iter_02_results.json'
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")
        print(f"Experiment completed in {time.time() - start_time:.2f} seconds")
        
        # Final summary
        if len(results) >= 50:
            print(f"\n✓ SUCCESS: Collected {len(results)} samples (≥50 required)")
        else:
            print(f"\n✗ WARNING: Only collected {len(results)} samples (<50 required)")
            
        if analysis.get('meets_threshold', False):
            print("✓ SUCCESS: Ordering effect exceeds threshold (>0.15)")
        else:
            effect = analysis.get('ordering_effect', 0)
            print(f"✗ Ordering effect ({effect:.3f}) does not exceed threshold (0.15)")
    
    except Exception as e:
        print(f"Experiment failed with error: {e}")
        # Save error information
        error_data = {
            'error': str(e),
            'execution_time_seconds': time.time() - start_time
        }
        
        os.makedirs('/Users/kumacmini/cost-aware-research-search/results', exist_ok=True)
        output_path = '/Users/kumacmini/cost-aware-research-search/results/iter_02_results.json'
        with open(output_path, 'w') as f:
            json.dump(error_data, f, indent=2)

if __name__ == "__main__":
    main()