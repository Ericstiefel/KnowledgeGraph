"""
The Clustering process of KG generation is included here
If the general concept of the process is fuzzy or inclear, feel free to check out info/concept.pdf, hopefully it's able to clear some things up

Steps:

    (repeat 1-3 until pass n times without a confirmed merge)
    (1) Entities list passed into LLM, generates 1 cluster
    (2) Classification model confirms or denies cluster
    (3) If accepted, another labeling model assigns label to cluster

    (repeat until no entities remain)
    (4) Remaining entities are checked batch by batch, to be added to a cluster

"""

from models.Gemini import prompt 
from models.GenerateLabel import SeqToOne 
from models.LLMJudge import Judge 

def extract_cluster(candidates, instructions):
    unique_sorted_candidates = sorted(list(set(candidates)))
    prompt_text = f"{instructions}\nItems:\n{', '.join(unique_sorted_candidates)}\nExtracted Cluster:"
    response = prompt(prompt_text) 

    
    potential_cluster_elements = [e.strip() for e in response.split(',') if e.strip()]
    
    
    actual_cluster = [item for item in potential_cluster_elements if item in candidates]
    return actual_cluster


def cluster_items(item_list, instructions, llm_as_judge, llm_judge_prompt, max_failed_loops=3, batch_size_for_recheck=5): 
    remaining_items = set(item_list)
    clusters = [] 
    cluster_labels = [] 

    failed_extraction_attempts = 0

  
    while failed_extraction_attempts < max_failed_loops and len(remaining_items) >= 2 : 
        current_candidate_list = list(remaining_items)
       
        proposed_cluster_items = extract_cluster(current_candidate_list, instructions)
       

        if not proposed_cluster_items or len(proposed_cluster_items) < 2: 
            failed_extraction_attempts += 1
            continue

        response = False

        if llm_as_judge:
            response = prompt(llm_judge_prompt + str(proposed_cluster_items))
        else:
            response = Judge(proposed_cluster_items)

        
        if response: 
            clusters.append(proposed_cluster_items)
            label = SeqToOne(proposed_cluster_items) 
            cluster_labels.append(label)
            
            for item in proposed_cluster_items:
                remaining_items.discard(item)
            failed_extraction_attempts = 0 
        else:

            failed_extraction_attempts += 1

    unclustered_list = list(remaining_items)
    items_added_in_pass = True 

    while items_added_in_pass and unclustered_list and clusters:
        items_added_in_pass = False
        items_to_remove_from_unclustered = []

        for i in range(0, len(unclustered_list), batch_size_for_recheck):
            batch_to_check = unclustered_list[i:i + batch_size_for_recheck]
            
            for item_to_add in batch_to_check:
                if item_to_add in items_to_remove_from_unclustered: 
                    continue

                added_to_a_cluster = False
                for cluster_idx, existing_cluster in enumerate(clusters):

                    potential_extended_cluster = existing_cluster + [item_to_add]
                    
                    if Judge(potential_extended_cluster):
                        clusters[cluster_idx].append(item_to_add)
                        
                        items_to_remove_from_unclustered.append(item_to_add)
                        items_added_in_pass = True
                        added_to_a_cluster = True
                        break 
        
        if items_to_remove_from_unclustered:
            new_unclustered_list = [item for item in unclustered_list if item not in items_to_remove_from_unclustered]
            unclustered_list = new_unclustered_list

    formatted_clusters = [{"label": cluster_labels[i], "items": clusters[i]} for i in range(len(clusters))]
    return formatted_clusters, unclustered_list


def extract_entities_from_triples(triples):
    subjects = [s for s, _, _ in triples]
    objects = [o for _, _, o in triples]
    entity_list = list(set(subjects + objects))
    return entity_list


def cluster_entities_from_triples(triples, entity_instructions, llm_as_judge = False, llm_judge_prompt = '', max_failed_loops_cluster_extraction=3, batch_size_for_recheck=5):
    entity_list = extract_entities_from_triples(triples)

    if not entity_list:
        return {
            "entity_clusters": [],
            "leftover_entities": []
        }
        

    entity_clusters, leftover_entities = cluster_items(
        entity_list,
        instructions=entity_instructions,
        llm_as_judge=llm_as_judge,
        llm_judge_prompt=llm_judge_prompt,
        max_failed_loops=max_failed_loops_cluster_extraction,
        batch_size_for_recheck=batch_size_for_recheck
    )
    return {
        "entity_clusters": entity_clusters,
        "leftover_entities": leftover_entities
    }
