from models.EntityDetection import BiLSTM_CRF
from models.Gemini import prompt
import torch
from transformers import AutoTokenizer
import ast
import re
import os

"""

The Generation process of KG generation is included here
If the general concept of the process is fuzzy or inclear, feel free to check out info/concept.pdf, hopefully it's able to clear some things up


Steps to the Generation segment:

    (1) Extract Entities from text
    (2) Invoke LLM to produce (subject, predicate, object) triples
"""

def getEntities(text: str, llm=False, llm_entities_prompt=''): 
    if llm:
        return prompt(llm_entities_prompt + text)
    else:

        tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        label_list = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']
        id2label = {i: label for i, label in enumerate(label_list)}
        vocab_size = tokenizer.vocab_size
        model = BiLSTM_CRF(vocab_size, embedding_dim=128, hidden_dims=256, num_tags=len(label_list))
        try:
            model.load_state_dict(torch.load("C:\\Users\\15169\\Documents\\KnowledgeGraph\\models\\EntityDetection.pt"))
        except FileNotFoundError:
            print(f"ERROR: EntityDetection model not found.")
            return []
        except Exception as e:
            print(f"Error loading EntityDetection model: {e}")
            return []
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        tokens = tokenizer.tokenize(text)
        if not tokens:
            print("Warning: No tokens generated for sentence:", text)
            return []
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        max_len = 512 
        pad_token_id = tokenizer.pad_token_id

        if len(input_ids) > max_len:
            input_ids_padded = input_ids[:max_len]
            mask = [1] * max_len
        else:
            input_ids_padded = input_ids + [pad_token_id] * (max_len - len(input_ids))
            mask = [1] * len(input_ids) + [0] * (max_len - len(input_ids))

        input_tensor = torch.tensor([input_ids_padded], dtype=torch.long).to(device)
        mask_tensor = torch.tensor([mask], dtype=torch.bool).to(device)
        with torch.no_grad():
            predictions = model(input_tensor, mask=mask_tensor)
            if not predictions or not predictions[0]:
                print("Warning: Model returned no predictions for text:", text)
                return []
            predicted_tags_for_first_item = predictions[0]
            predicted_labels = predicted_tags_for_first_item[:len(tokens)]
        decoded_labels = [id2label.get(tag, 'O') for tag in predicted_labels]
        binary_labels = [0 if tag == 'O' else 1 for tag in decoded_labels]
        full_words, current_word, current_label = [], "", 0
        for token, label in zip(tokens, binary_labels):
            if token.startswith("##"): current_word += token[2:]
            else:
                if current_word: full_words.append((current_word, current_label))
                current_word, current_label = token, 0
            if label == 1: current_label = 1
        if current_word: full_words.append((current_word, current_label))
        return [word for word, word_label_val in full_words if word_label_val == 1]


def parse_llm_output_to_triples(llm_string: str) -> list:
    if not llm_string or not llm_string.strip():
        print("Warning: LLM returned empty string.")
        return []

    parsed_triples = []
    
    matches = re.findall(r"(?:\d+\.\s*|\*\s*\*\*)?\(([^,]+?),([^,]+?),([^)]+?)\)(?:\*\*)?", llm_string)

    for match_tuple in matches:
        s_raw, p_raw, o_raw = match_tuple
        
        s = s_raw.strip().replace("**", "").strip()
        p = p_raw.strip().replace("**", "").strip()
        o = o_raw.strip().replace("**", "").strip() 

        note_starters_regex = r"\s*(?:\)\*\*|\))?\s*\*\s*\(\s*Note:|\s*\(\s*Note:|\s*Note:"
        
        note_match = re.search(note_starters_regex, o)
        if note_match:
            o = o[:note_match.start()].strip()
        
        o = o.strip().replace("**", "").strip().strip("'\"")

        if s and p and o:
            parsed_triples.append((s, p, o))
        else:
            print(f"Warning: Skipped a malformed triple after cleaning: raw=({s_raw}, {p_raw}, {o_raw}), cleaned=({s},{p},{o})")
    
    if parsed_triples:
        print(f"Info: Successfully parsed {len(parsed_triples)} triples using primary regex.")
        return parsed_triples
    print("Info: Primary regex failed or yielded no results, trying fallback methods...")
    if not parsed_triples:
        cleaned_string_for_ast = llm_string.strip()
        if cleaned_string_for_ast.startswith("[") and cleaned_string_for_ast.endswith("]"):
            try:
                triples_list_from_ast = ast.literal_eval(cleaned_string_for_ast)
                if isinstance(triples_list_from_ast, list): 
                    parsed_triples = [t for t in triples_list_from_ast if isinstance(t, tuple) and len(t) == 3]
                    if parsed_triples:
                        print(f"Info: Parsed {len(parsed_triples)} triples using ast.literal_eval fallback.")
            except Exception as e:
                print(f"Warning: ast.literal_eval fallback failed: {e}")

    if not parsed_triples:
         print(f"CRITICAL WARNING: All parsing attempts failed. Raw output (first 300 chars): {llm_string[:300]}")
    
    return parsed_triples

def generate(text: str, triples_prompt: str, use_llm_for_detection=False, llm_detection_prompt='') -> list:
    entities: list = getEntities(text, use_llm_for_detection, llm_detection_prompt)
    print('Entities Identified: ', entities)
    input_to_llm = triples_prompt + '\nentities: ' + str(entities) + '\nsource text: ' + text
    llm_response_str = prompt(input_to_llm)
    print(f"--- Response ---\n{llm_response_str}\n------------------------\n")
    parsed_triples = parse_llm_output_to_triples(llm_response_str)
    return parsed_triples

if __name__ == '__main__':
    triples_prompt_example = (
        "Return a list of (subject, predicate, object). Extract subject-predicate-object "
        "triples from the assistant message. A predicate (1 word) defines the relationship "
        "between the subject and object. Relationship may be fact or sentiment based on "
        "assistant’s message. Subject and object are entities. Entities provided are from "
        "the assistant message and prior conversation history, though you may not need all of them. "
        "This is for an extraction task, please be thorough, accurate, and faithful to the "
        "reference text. The following is a list of entities, followed by the source text, denoted as so."
    )
    example_text_main_issue = "Jack went to the mall to see Isabella, his girlfriend. Together they found Joseph's mom waiting for them there."

    print(f"--- Generating triples for (generate.py __main__): ---\n'{example_text_main_issue}'\n-----------------------------------------------------\n")
    generated_triples_result = generate(example_text_main_issue, triples_prompt_example)
    print(f"\n--- Parsed Generated Triples (from generate.py __main__) ---\n{generated_triples_result}\n----------------------------------------------------------\n")