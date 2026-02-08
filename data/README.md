# Data Sources

PRISM's large-scale knowledge integration uses three external data sources.

## Sources

### ConceptNet 5.7
- **URL**: https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz
- **Size**: ~600MB compressed
- **Facts**: ~3M (English, confidence ≥ 2.0)
- **Relations**: IS-A, HAS, CAN, CAUSES, PART-OF, USED-FOR, LOCATED-AT, HAS-PROPERTY, MADE-OF, RELATED-TO

### WordNet (via NLTK)
- **Source**: NLTK WordNet corpus
- **Size**: ~12MB (auto-downloaded)
- **Facts**: ~200K relations
- **Relations**: IS-A (hypernyms), HAS-TYPE (hyponyms), HAS-PART (meronyms), PART-OF (holonyms), SIMILAR-TO (synonyms)

### Simple English Wikipedia
- **URL**: https://dumps.wikimedia.org/simplewiki/latest/simplewiki-latest-pages-articles.xml.bz2
- **Size**: ~250MB compressed
- **Facts**: ~500K from 50K articles
- **Extraction**: Uses spaCy-based FactExtractor for SVO triple extraction

## Directory Structure

```
data/
├── README.md          # This file
├── cache/             # Downloaded and cached data
│   ├── conceptnet-assertions-5.7.0.csv.gz
│   ├── conceptnet_facts.json.gz
│   ├── simplewiki-latest-pages-articles.xml.bz2
│   └── simplewiki_facts.json.gz
├── loaders/           # Loader modules
│   ├── conceptnet_loader.py
│   ├── wordnet_loader.py
│   ├── simplewiki_loader.py
│   └── knowledge_integrator.py
├── knowledge_base.json.gz  # Merged facts from all sources
└── trained_memory.npz      # Trained VSA memory
```

## Training

```bash
# Quick test with sample data (no downloads)
python scripts/train_knowledge.py --sample

# Train on WordNet only (fast, ~2 min)
python scripts/train_knowledge.py --sources wordnet

# Train on ConceptNet + WordNet (~10 min)
python scripts/train_knowledge.py --sources conceptnet,wordnet --cache

# Full training (~45 min)
python scripts/train_knowledge.py --sources conceptnet,wordnet,simplewiki
```

## Using Trained Memory

```bash
# Start PRISM with pre-trained knowledge
python -m prism --load-knowledge data/trained_memory
```
