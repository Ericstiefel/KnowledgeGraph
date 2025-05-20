from generate import generate
from aggregation import aggregate_triples
from cluster import cluster_entities_from_triples
from ConvertToText.main import getData



"""
This file is comprised of the main loop, gathering the generation, aggregation, and clustering processes.
Very general description of the whole KG generating process can be found in info/concept.pdf

"""


def run(
        text: str, 
        use_llm_for_detection = False,
        use_llm_as_judge = False
        ):
    triples_prompt_strict = (
    "Extract subject-predicate-object triples from the 'source text' below. "
    "STRICTLY follow these rules:\n"
    "1. Output format: Return ONLY a Python-style list of tuples. For example: `[('subject1', 'predicate1', 'object1'), ('subject2', 'predicate2', 'object2')]`.\n"
    "2. NO EXTRA TEXT: Do NOT include any explanations, comments, notes, apologies, or any conversational text before or after the list.\n"
    "3. Predicate: The predicate MUST be a single lowercase word representing the core relationship.\n"
    "4. Entities: Subjects and objects are entities. You are given a list of 'entities' to primarily focus on. "
    "   If a triple requires an entity not in the provided list to be factually correct based on the source text, you MAY include it. "
    "   However, prioritize using entities from the provided list where appropriate.\n"
    "5. Accuracy: Be accurate and faithful to the relationships explicitly stated or very directly implied in the source text.\n"
    "6. Completeness: Extract all relevant triples based on the text.\n"
    "\n"
)

    
    clustering_entity_prompt = "Find ONE cluster of related entities from this list. A" \
    " cluster should contain entities that are the same in" \
    " meaning, with different:" \
    " - tenses" \
    " - plural forms" \
    " - stem forms" \
    " - upper/lower cases" \
    " Or entities with close semantic meanings." \
    " Return only if you find entities that clearly belong" \
    " together." \
    " If you can’t find a clear cluster, return an empty list."

    use_llm_as_judge_prompt = ("Verify if these entities belong in the same cluster."
    "A cluster should contain entities that are the same in"
    "meaning, with different:"
    "- tenses"
    "- plural forms"
    "- stem forms"
    "- upper/lower cases"
    "Or entities with close semantic meanings."
    "Return the entities that you are confident belong together"
    "as a single cluster."
    "If you’re not confident, return an empty list."
    "Prompt for clustering edges"
    "Find ONE cluster of closely related predicates from this"
    "list."
    "A cluster should contain predicates that are the same in"
    "meaning, with different:"
    "- tenses"
    "- plural forms"
    "- stem forms"
    "- upper/lower cases"
    "Predicates are the relations between subject and object"
    "entities. Ensure that the predicates in the same cluster"
    "have very close semantic meanings to describe the relation"
    "between the same subject and object entities."
    "Return only if you find predicates that clearly belong"
    "together."
    "If you can’t find a clear cluster, return an empty list.")


    if text.lower().endswith('.pdf') or text.lower().endswith('.pptx'):
        words = getData(text)
    else:
        words = text

    generated_triples = generate(words, triples_prompt_strict, use_llm_for_detection)

    aggregated_triples = aggregate_triples(generated_triples)

    clustered_triples = cluster_entities_from_triples(aggregated_triples, clustering_entity_prompt, llm_as_judge=use_llm_as_judge, llm_judge_prompt=use_llm_as_judge_prompt)

    return clustered_triples

if __name__ == '__main__':
    passage = 'Jack went to the mall to see Isabella, his girlfriend. Together they found Joseph\'s mom waiting for them there.'
    print(run(passage))