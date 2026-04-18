# Medical RAG Data Sources Plan

## Goal
Build a broad, trustworthy medical RAG corpus using authoritative public-health and clinical sources, then use QA datasets for evaluation (not primary truth).

## Source Tiers

### Tier 1 (Primary RAG sources: ingest first)
- WHO Publications: https://www.who.int/publications
- WHO IRIS repository: https://iris.who.int
- NLM data distribution: https://www.nlm.nih.gov/databases/download/data_distrib_main.html
- MedlinePlus XML feeds: https://medlineplus.gov/xml.html

Use these as the main knowledge base for patient-facing factual responses.

### Tier 2 (High-value, but often access/licensing/manual steps)
- PubMed Open Access Subset: https://www.ncbi.nlm.nih.gov/pmc/tools/openftlist/
- MIMIC-IV portal: https://physionet.org/content/mimiciv
- MIMIC code: https://github.com/MIT-LCP/mimic-code
- DrugBank releases: https://go.drugbank.com/releases/latest
- DrugBank org tools: https://github.com/drugbank

These may require credentialing, large storage, or strict usage terms.

### Tier 3 (Evaluation/benchmark datasets)
- PubMedQA repo: https://github.com/pubmedqa/pubmedqa
- Symptom-disease Kaggle: https://www.kaggle.com/datasets/kaushil268/disease-prediction-using-machine-learning
- Symptom-disease mirror: https://github.com/Ankit152/Medical-Symptom-Dataset

Use for QA benchmarking and retrieval tests; do not treat as sole clinical truth.

## Ingestion Strategy
1. Prefer direct guideline PDFs and official XML/JSON feeds over random web pages.
2. Store source metadata per chunk:
   - `source_name`
   - `source_url`
   - `category`
   - `publication_year` (if available)
   - `doc_type` (`guideline`, `dataset`, `qa_benchmark`, etc.)
3. Keep only trustworthy/authoritative medical content in primary index.
4. Keep benchmark datasets in a separate index/namespace.

## Indexing Layout (Recommended)
- `medical-guidelines` (WHO, NIH/NLM, CDC, specialty guidelines)
- `medical-research` (PubMed OA subset selected summaries)
- `medical-eval` (PubMedQA + symptom datasets)

If using one Pinecone index, use namespaces with the names above.

## Operational Notes
- MIMIC datasets are large and access-controlled; ingest only approved derived docs.
- DrugBank has licensing requirements; verify terms before ingestion.
- Kaggle datasets may need API auth and quality filtering.

## Immediate Next Steps
1. Replace landing-page URLs in `server/knowledge_sources.json` with direct PDF/XML endpoints.
2. Run bulk ingestion in batches:
   - `python server\\scripts\\bulk_ingest_sources.py --catalog server\\knowledge_sources.json --api-url http://127.0.0.1:8000 --limit 20`
3. Add metadata-aware retrieval filtering by specialty and source reliability.
4. Add an evaluation set (100+ questions) and track answer grounding rate.
