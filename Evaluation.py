from datasets import load_dataset
from main import run
from models.Gemini import prompt 
import time
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import numpy as np
import collections
import re
import string

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction, ground_truths):
    """Computes F1 score between a prediction and a list of ground truths."""
    if not prediction or not ground_truths:
        return 0
    
    normalized_prediction = normalize_answer(prediction)
    prediction_tokens = normalized_prediction.split()
    
    max_f1 = 0
    for gt in ground_truths:
        normalized_ground_truth = normalize_answer(gt)
        ground_truth_tokens = normalized_ground_truth.split()
        
        if not prediction_tokens or not ground_truth_tokens:
            if not prediction_tokens and not ground_truth_tokens:
                current_f1 = 1.0
            else: 
                current_f1 = 0.0
            max_f1 = max(max_f1, current_f1)
            continue

        common = collections.Counter(prediction_tokens) & collections.Counter(ground_truth_tokens)
        num_same = sum(common.values())

        if num_same == 0:
            current_f1 = 0.0
            max_f1 = max(max_f1, current_f1)
            continue

        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        current_f1 = (2 * precision * recall) / (precision + recall)
        max_f1 = max(max_f1, current_f1)
        
    return max_f1

def exact_match_score(prediction, ground_truths):
    if not prediction or not ground_truths:
        return 0
    prediction_normalized = normalize_answer(prediction)
    for gt in ground_truths:
        gt_normalized = normalize_answer(gt)
        if prediction_normalized == gt_normalized:
            return 1
    return 0

def answer_from_triples(
    triples: list,
    context: str,
    question: str,
    use_llm_to_synthesize_answer: bool,
    llm_synthesis_func
    ) -> str:
    if not triples or not isinstance(triples, list) or not all(isinstance(t, tuple) for t in triples):
        if use_llm_to_synthesize_answer:
            prompt_text = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
            return llm_synthesis_func(prompt_text)
        else:
            return "No triples generated for non-LLM synthesis."

    if use_llm_to_synthesize_answer:
        triples_str = "\n".join([f"- {t[0]} {t[1]} {t[2]}" for t in triples if len(t)==3])
        prompt_text = (
            f"Based on the following context and extracted structured information (triples), "
            f"please answer the question.\n\n"
            f"Context: {context}\n\n"
            f"Extracted Triples:\n{triples_str if triples_str else 'No relevant triples extracted.'}\n\n"
            f"Question: {question}\n\n"
            f"Answer:"
        )
        return llm_synthesis_func(prompt_text)
    else:
        q_words = set(normalize_answer(question).replace('?', '').split())
        for tpl in triples:
            if not (isinstance(tpl, tuple) and len(tpl) == 3): continue
            subj, pred, obj = map(str, tpl)
            if q_words.intersection(set(normalize_answer(subj).split()) | set(normalize_answer(obj).split())):
                return f"Found relevant triple (non-LLM): {subj} {pred} {obj}."
        return "Could not find a direct answer in triples using simple non-LLM method."

def baseline_architecture(context: str, question: str) -> str:
    try:
        prompt_text = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
        response = prompt(prompt_text)
        return response if response else "Error: No response from baseline LLM"
    except Exception as e:
        return f"Error executing baseline: {e}"

def lightweight_architecture(context: str, question: str) -> str:
    try:
        triples = run(text=context, use_llm_for_detection=False, use_llm_as_judge=False)
        answer = answer_from_triples(
            triples, context, question,
            use_llm_to_synthesize_answer=True,
            llm_synthesis_func=prompt 
        )
        return answer
    except Exception as e:
        return f"Error executing lightweight: {e}"


def heavyweight_architecture(context: str, question: str) -> str:
    try:
        triples = run(text=context, use_llm_for_detection=True, use_llm_as_judge=True)
        answer = answer_from_triples(
            triples, context, question,
            use_llm_to_synthesize_answer=True, 
            llm_synthesis_func=prompt
        )
        return answer
    except Exception as e:
        return f"Error executing heavyweight: {e}"


if __name__ == '__main__':
    print("Loading SQuAD validation dataset")
    try:
        squad_dataset = load_dataset('squad', split='validation')
    except Exception as e:
        print(f"Failed to load SQuAD dataset: {e}")
        exit()

    squad_dataset = squad_dataset.shuffle(seed=8)

    MAX_EXAMPLES = 1000

    if MAX_EXAMPLES and MAX_EXAMPLES < len(squad_dataset):
        squad_eval_subset = squad_dataset.select(range(MAX_EXAMPLES))
        print(f"Using a subset of {MAX_EXAMPLES} examples for this run.")
    else:
        squad_eval_subset = squad_dataset
        print(f"Using full validation set ({len(squad_eval_subset)} examples) for this run.")

    results_collection = []
    architectures = {
        "Baseline": baseline_architecture,
        "Lightweight": lightweight_architecture,
        "Heavyweight": heavyweight_architecture
    }

    print("\nRunning evaluation on SQuAD examples...")
    for example in tqdm(squad_eval_subset, desc="Evaluating Architectures"):
        context = example['context']
        question = example['question']
        ground_truths = example['answers']['text']
        example_id = example['id']
        text_length = len(context)


        current_example_results = {
            'id': example_id,
            'context': context, 
            'question': question,
            'text_length': text_length,
            'ground_truths': ground_truths,
            'predictions': {}
        }

        for arch_name, arch_func in architectures.items():
            start_time = time.time()
            pred_answer = arch_func(context, question)
            time_taken = time.time() - start_time
            
            em_accuracy = exact_match_score(pred_answer, ground_truths)
            f1 = f1_score(pred_answer, ground_truths)


            current_example_results['predictions'][arch_name] = {
                'answer': pred_answer,
                'time_seconds': time_taken,
                'accuracy_em': em_accuracy,
                'f1_score': f1
            }
        results_collection.append(current_example_results)

    print(f"\n--- Evaluation completed. Processed {len(results_collection)} examples. ---")

    results_output_filename = 'squad_evaluation_results_with_metrics.json'
    print(f"Saving detailed results to {results_output_filename}...")
    try:
        with open(results_output_filename, 'w', encoding='utf-8') as f:
            json.dump(results_collection, f, indent=4, ensure_ascii=False)
        print("Results saved successfully.")
    except Exception as e:
        print(f"An error occurred while saving results: {e}")

    if not results_collection:
        print("No results to plot.")
        exit()

    plot_data = collections.defaultdict(lambda: {'text_lengths': [], 'accuracies': [], 'runtimes': [], 'f1_scores': []})

    for item in results_collection:
        text_len = item['text_length']
        for arch_name in architectures.keys():
            if arch_name in item['predictions']:
                plot_data[arch_name]['text_lengths'].append(text_len)
                plot_data[arch_name]['accuracies'].append(item['predictions'][arch_name]['accuracy_em'])
                plot_data[arch_name]['runtimes'].append(item['predictions'][arch_name]['time_seconds'])
                plot_data[arch_name]['f1_scores'].append(item['predictions'][arch_name]['f1_score'])


    all_text_lengths = [item['text_length'] for item in results_collection]
    if not all_text_lengths:
        print("No text lengths recorded for plotting.")
        exit()

    min_len, max_len = min(all_text_lengths), max(all_text_lengths)
    num_bins = min(10, len(set(all_text_lengths))) if len(set(all_text_lengths)) > 1 else 1
    if num_bins == 1 and len(all_text_lengths) > 0 : 
         bins = np.array([min_len, max_len +1e-5]) 
    elif max_len == min_len :
        if max_len == 0:
            bins = np.linspace(0, 1, num_bins +1 if num_bins > 0 else 2)
        else:
            bins = np.linspace(min_len - 0.5 * abs(min_len) if min_len != 0 else -0.5, 
                               max_len + 0.5 * abs(max_len) if max_len != 0 else 0.5, 
                               num_bins + 1 if num_bins > 0 else 2)
    else:
        bins = np.linspace(min_len, max_len, num_bins + 1 if num_bins > 0 else 2)

    if len(bins) <= 1: 
        bins = np.array([min_len, max_len + 1e-5]) 

    bin_centers = (bins[:-1] + bins[1:]) / 2 if len(bins) > 1 else np.array([(min_len + max_len)/2])


    binned_accuracies = collections.defaultdict(lambda: np.full(num_bins, np.nan))
    binned_runtimes = collections.defaultdict(lambda: np.full(num_bins, np.nan))
    binned_f1_scores = collections.defaultdict(lambda: np.full(num_bins, np.nan)) 

    for arch_name in architectures.keys():
        if not plot_data[arch_name]['text_lengths']: continue

        arch_text_lengths = np.array(plot_data[arch_name]['text_lengths'])
        arch_accuracies = np.array(plot_data[arch_name]['accuracies'])
        arch_runtimes = np.array(plot_data[arch_name]['runtimes'])
        arch_f1_scores = np.array(plot_data[arch_name]['f1_scores']) 


        for i in range(num_bins):
            if len(bins) <= 1 : 
                 bin_mask = (arch_text_lengths >= bins[0]) & (arch_text_lengths <= bins[0]) 
            elif i == num_bins - 1: 
                bin_mask = (arch_text_lengths >= bins[i]) & (arch_text_lengths <= bins[i+1])
            else:
                bin_mask = (arch_text_lengths >= bins[i]) & (arch_text_lengths < bins[i+1])
            
            if np.any(bin_mask):
                binned_accuracies[arch_name][i] = np.mean(arch_accuracies[bin_mask])
                binned_runtimes[arch_name][i] = np.mean(arch_runtimes[bin_mask])
                binned_f1_scores[arch_name][i] = np.mean(arch_f1_scores[bin_mask]) 
    
    plt.style.use('seaborn-v0_8-whitegrid')

    plt.figure(figsize=(12, 7))
    for arch_name in architectures.keys():
        if not np.all(np.isnan(binned_accuracies[arch_name])):
            plt.plot(bin_centers, binned_accuracies[arch_name], marker='o', linestyle='-', label=f'{arch_name} EM')
    plt.xlabel('Text Length (Number of Characters in Context)')
    plt.ylabel('Average Exact Match Score')
    plt.title('Accuracy (EM) vs. Text Length by Architecture')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    accuracy_plot_filename = 'accuracy_vs_text_length.png'
    try:
        plt.savefig(accuracy_plot_filename)
        print(f"Accuracy plot saved to {accuracy_plot_filename}")
    except Exception as e:
        print(f"Error saving accuracy plot: {e}")
    plt.show(block=False)

    plt.figure(figsize=(12, 7))
    for arch_name in architectures.keys():
        if not np.all(np.isnan(binned_runtimes[arch_name])):
            plt.plot(bin_centers, binned_runtimes[arch_name], marker='s', linestyle='--', label=f'{arch_name} Runtime')
    plt.xlabel('Text Length (Number of Characters in Context)')
    plt.ylabel('Average Runtime (seconds)')
    plt.title('Runtime vs. Text Length by Architecture')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    runtime_plot_filename = 'runtime_vs_text_length.png'
    try:
        plt.savefig(runtime_plot_filename)
        print(f"Runtime plot saved to {runtime_plot_filename}")
    except Exception as e:
        print(f"Error saving runtime plot: {e}")
    plt.show(block=False)


    plt.figure(figsize=(12, 7))
    for arch_name in architectures.keys():
        if not np.all(np.isnan(binned_f1_scores[arch_name])): 
            plt.plot(bin_centers, binned_f1_scores[arch_name], marker='^', linestyle=':', label=f'{arch_name} F1')
    plt.xlabel('Text Length (Number of Characters in Context)')
    plt.ylabel('Average F1 Score')
    plt.title('F1 Score vs. Text Length by Architecture')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    f1_plot_filename = 'f1_score_vs_text_length.png'
    try:
        plt.savefig(f1_plot_filename)
        print(f"F1 score plot saved to {f1_plot_filename}")
    except Exception as e:
        print(f"Error saving F1 score plot: {e}")
    plt.show() 

    print("\nPlotting complete. Check for .png files in the script directory.")
    print("Note: For more reliable and smoother graphs, run with a larger MAX_EXAMPLES.")