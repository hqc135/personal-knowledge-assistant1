from collections import Counter
import json
from statistics import mean, median
import sys
from pathlib import Path

# Ensure project root is on sys.path when running this script directly
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
	sys.path.insert(0, str(ROOT))

from kg_store import load_triples

triples = load_triples()
summary = {}
summary['total_triples'] = len(triples)
chunk_ids = [t.get('chunk_id') for t in triples if t.get('chunk_id')]
summary['unique_chunks'] = len(set(chunk_ids))
per_chunk = Counter(chunk_ids)
summary['avg_triples_per_chunk'] = mean(per_chunk.values()) if per_chunk else 0
summary['median_triples_per_chunk'] = median(per_chunk.values()) if per_chunk else 0

# entities
heads = [t['head'] for t in triples if t.get('head')]
tails = [t['tail'] for t in triples if t.get('tail')]
ents = heads + tails
ent_counts = Counter(ents)
summary['unique_entities'] = len(ent_counts)
summary['top_entities'] = ent_counts.most_common(20)

# entity length stats
lengths = [len(e) for e in ent_counts]
summary['entity_len_min'] = min(lengths) if lengths else 0
summary['entity_len_max'] = max(lengths) if lengths else 0
summary['entity_len_mean'] = mean(lengths) if lengths else 0

# single-occurrence entities
summary['entities_single_occurrence'] = sum(1 for c in ent_counts.values() if c==1)
summary['entities_single_occurrence_pct'] = summary['entities_single_occurrence'] / summary['unique_entities'] if summary['unique_entities'] else 0

# chars in names
import re
non_alnum = sum(1 for e in ent_counts if re.search(r'[^\w\u4e00-\u9fff]', e))
summary['entities_with_non_alnum'] = non_alnum
summary['entities_with_non_alnum_pct'] = non_alnum / summary['unique_entities'] if summary['unique_entities'] else 0

# output
print(json.dumps(summary, ensure_ascii=False, indent=2))
