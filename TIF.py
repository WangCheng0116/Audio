import argparse
import json

def evaluate_response(gt, response):
    return gt.lower() in response.lower()

def get_all_gt(data):
    return {entry['gt'].lower() for entry in data}

def analyze_json(file_path):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        all_gt = get_all_gt(data)
        
        # Initialize statistical counts for accuracy
        stats = {
            'faithful': {'correct': 0, 'total': 0},
            'adversarial': {'correct': 0, 'total': 0},
            'irrelevant': {'correct': 0, 'total': 0},
            'neutral': {'correct': 0, 'total': 0}
        }
        
        # Initialize influence rate counters
        influence = {
            'faithful': {'wrong_to_correct': 0, 'total_neutral_wrong': 0},
            'adversarial': {'correct_to_wrong': 0, 'total_neutral_correct': 0},
            'irrelevant': {'flipped': 0, 'total': 0}
        }
        
        for entry in data:
            gt = entry['gt']
            
            # First evaluate the neutral response as baseline
            neutral_response = entry.get('neutral_response', '')
            neutral_correct = evaluate_response(gt, neutral_response) and any(gt_item in neutral_response.lower() for gt_item in all_gt)
            
            # Process each response type
            for response_type in ['faithful', 'adversarial', 'irrelevant', 'neutral']:
                response = entry.get(f'{response_type}_response', '')
                stats[response_type]['total'] += 1
                
                # Check if at least one element in all_gt appears in response
                is_correct = evaluate_response(gt, response) and any(gt_item in response.lower() for gt_item in all_gt)
                
                if is_correct:
                    stats[response_type]['correct'] += 1
                
                # Calculate influence metrics (comparing with neutral baseline)
                if response_type == 'faithful':
                    if not neutral_correct:  # neutral was wrong
                        influence['faithful']['total_neutral_wrong'] += 1
                        if is_correct:  # faithful is correct
                            influence['faithful']['wrong_to_correct'] += 1
                
                elif response_type == 'adversarial':
                    if neutral_correct:  # neutral was correct
                        influence['adversarial']['total_neutral_correct'] += 1
                        if not is_correct:  # adversarial is wrong
                            influence['adversarial']['correct_to_wrong'] += 1
                
                elif response_type == 'irrelevant':
                    influence['irrelevant']['total'] += 1
                    if neutral_correct != is_correct:  # flipped result
                        influence['irrelevant']['flipped'] += 1
        
        # Calculate accuracy
        results = {}
        for key in stats:
            correct = stats[key]['correct']
            total = stats[key]['total']
            accuracy = correct / total * 100 if total > 0 else 0
            results[key] = {
                'accuracy': round(accuracy, 2),
                'correct': correct,
                'total': total
            }
        
        # Calculate influence rates
        influence_rates = {}
        
        # Faithful: wrong -> correct rate
        wrong_to_correct = influence['faithful']['wrong_to_correct']
        total_neutral_wrong = influence['faithful']['total_neutral_wrong']
        faithful_influence = wrong_to_correct / total_neutral_wrong * 100 if total_neutral_wrong > 0 else 0
        influence_rates['faithful'] = {
            'influence_rate': round(faithful_influence, 2),
            'wrong_to_correct': wrong_to_correct,
            'total_neutral_wrong': total_neutral_wrong
        }
        
        # Adversarial: correct -> wrong rate
        correct_to_wrong = influence['adversarial']['correct_to_wrong']
        total_neutral_correct = influence['adversarial']['total_neutral_correct']
        adversarial_influence = correct_to_wrong / total_neutral_correct * 100 if total_neutral_correct > 0 else 0
        influence_rates['adversarial'] = {
            'influence_rate': round(adversarial_influence, 2),
            'correct_to_wrong': correct_to_wrong,
            'total_neutral_correct': total_neutral_correct
        }
        
        # Irrelevant: flip rate
        flipped = influence['irrelevant']['flipped']
        total = influence['irrelevant']['total']
        irrelevant_influence = flipped / total * 100 if total > 0 else 0
        influence_rates['irrelevant'] = {
            'influence_rate': round(irrelevant_influence, 2),
            'flipped': flipped,
            'total': total
        }
        
        return results, influence_rates
    
    except FileNotFoundError:
        print(f"Error: File {file_path} not found!")
    except Exception as e:
        print(f"Error: An unknown error occurred: {e}")
    return None, None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze data in a JSON file')
    parser.add_argument('file_path', type=str, help='Path to the input JSON file')
    args = parser.parse_args()
    
    file_path = args.file_path
    results, influence_rates = analyze_json(file_path)
    
    if results is not None:
        print("Accuracy Results:")
        for key, value in results.items():
            print(f"\n{key.capitalize()} Response:")
            print(f"Accuracy: {value['accuracy']}%")
            print(f"Correct: {value['correct']}/{value['total']}")
        
        print("\nInfluence Rate Results:")
        
        print("\nFaithful Influence (Wrong → Correct):")
        print(f"Rate: {influence_rates['faithful']['influence_rate']}%")
        print(f"Count: {influence_rates['faithful']['wrong_to_correct']}/{influence_rates['faithful']['total_neutral_wrong']}")
        
        print("\nAdversarial Influence (Correct → Wrong):")
        print(f"Rate: {influence_rates['adversarial']['influence_rate']}%")
        print(f"Count: {influence_rates['adversarial']['correct_to_wrong']}/{influence_rates['adversarial']['total_neutral_correct']}")
        
        print("\nIrrelevant Influence (Flipped):")
        print(f"Rate: {influence_rates['irrelevant']['influence_rate']}%")
        print(f"Count: {influence_rates['irrelevant']['flipped']}/{influence_rates['irrelevant']['total']}")
