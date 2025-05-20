from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import re
import string
from models.GenerateLabel import SeqToOne
from collections import defaultdict

"""
The Aggregation process of KG generation is included here
If the general concept of the process is fuzzy or inclear, feel free to check out info/concept.pdf, hopefully it's able to clear some things up
"""

def normalize_entity(entity):
    entity = entity.lower()
    entity = re.sub(rf"[{string.punctuation}]", "", entity)
    return entity.strip()

def group_entities(entities, threshold=0.75):
    norm_entities = list(set([normalize_entity(e) for e in entities]))
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(norm_entities)

    sim_matrix = cosine_similarity(embeddings)
    
    n = len(norm_entities)
    visited = set()
    clusters = []
    
    for i in range(n):
        if i in visited:
            continue
        cluster = [i]
        visited.add(i)
        for j in range(i + 1, n):
            if sim_matrix[i][j] >= threshold:
                cluster.append(j)
                visited.add(j)
        clusters.append(cluster)
    
    canonical_map = {}
    for cluster in clusters:
        members = [norm_entities[i] for i in cluster]
        canonical = max(members, key=len)
        for member in members:
            canonical_map[member] = canonical
    
    return canonical_map

def aggregate_triples(triples):
    entities = [subj for subj, _, _ in triples] + [obj for _, _, obj in triples]
    canonical_map = group_entities(entities)

    new_triples = []
    for s, p, o in triples:
        s_norm = normalize_entity(s)
        o_norm = normalize_entity(o)
        s_canon = canonical_map.get(s_norm, s_norm)
        o_canon = canonical_map.get(o_norm, o_norm)
        new_triples.append((s_canon, p, o_canon))
    return new_triples



def combineDuplicates(triples, model, id2word):
    pair_to_predicates = defaultdict(list)
    for subj, pred, obj in triples:
        pair_to_predicates[(subj, obj)].append(pred)
    combined_triples = []
    for (subj, obj), predicate_cluster in pair_to_predicates.items():
        if len(predicate_cluster) == 1:
            combined_triples.append((subj, predicate_cluster[0], obj))
        else:
            canonical_pred = SeqToOne(predicate_cluster, model, id2word)
            combined_triples.append((subj, canonical_pred, obj))

    return combined_triples

    
if __name__ == '__main__':
    triples = [
        ("Trump", "is headquartered in", "Cupertino"),
        ("trump", "headquartered_at", "Cupertino, California"),
        ("USA", "based in", "Cupertino"),
        ("America", "has office in", "Mountain View"),
        ("US", "headquarters at", "mountain view")
    ]

    aggregated = aggregate_triples(triples)
    for t in aggregated:
        print(t)
