"""
Benchmark: LSR vs Top-K vs MMR on HotpotQA

Downloads a 200-question subset of HotpotQA, embeds passages with
sentence-transformers (all-MiniLM-L6-v2), and compares retrieval methods
on redundancy, supporting-fact recall, and variance coverage.
"""

import sys
import os
import json
import time
import warnings

import numpy as np
from sklearn.decomposition import PCA

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.lsr import LocalSpectralRetrieval, TopKRetrieval, MMRRetrieval

NUM_QUESTIONS = 200
K = 5
THRESHOLD = 0.3
MMR_LAMBDA = 0.5
MODEL_NAME = "all-MiniLM-L6-v2"
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "benchmark_results.json")


def avg_pairwise_cosine(embeddings):
    if len(embeddings) < 2:
        return 0.0
    sim_matrix = embeddings @ embeddings.T
    n = len(sim_matrix)
    triu_indices = np.triu_indices(n, k=1)
    return float(np.mean(sim_matrix[triu_indices]))


def supporting_fact_recall(selected_indices, passage_titles, gold_titles):
    if len(gold_titles) == 0:
        return 1.0
    selected_titles = set(passage_titles[i] for i in selected_indices)
    covered = selected_titles & gold_titles
    return len(covered) / len(gold_titles)


def variance_coverage(selected_embeddings, full_embeddings):
    if len(selected_embeddings) < 2 or len(full_embeddings) < 2:
        return 0.0
    std_selected = np.std(selected_embeddings, axis=0)
    std_full = np.std(full_embeddings, axis=0)
    mask = std_full > 1e-9
    if mask.sum() == 0:
        return 0.0
    ratios = std_selected[mask] / std_full[mask]
    return float(np.mean(ratios))


def pc1_variance_ratio(embeddings):
    if len(embeddings) < 3:
        return 1.0
    pca = PCA(n_components=min(3, len(embeddings), embeddings.shape[1]))
    pca.fit(embeddings)
    return float(pca.explained_variance_ratio_[0])


def main():
    print('=' * 70)
    print('LOADING HOTPOTQA (validation[:200])')
    print('=' * 70)
    from datasets import load_dataset

    t0 = time.time()
    ds = load_dataset('hotpot_qa', 'distractor', split=f'validation[:{NUM_QUESTIONS}]')
    print(f'  Loaded {len(ds)} questions in {time.time() - t0:.1f}s')

    print()
    print('LOADING SENTENCE-TRANSFORMERS MODEL')
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_NAME)
    print(f'  Model: {MODEL_NAME}  (dim={model.get_sentence_embedding_dimension()})')

    print()
    print('EMBEDDING PASSAGES...')
    t0 = time.time()
    all_passages = []
    for qi, example in enumerate(ds):
        titles = example['context']['title']
        sents_per_passage = example['context']['sentences']
        for pi, (title, sents) in enumerate(zip(titles, sents_per_passage)):
            text = ' '.join(sents)
            all_passages.append((qi, pi, title, text))

    passage_texts = [p[3] for p in all_passages]
    passage_embeddings = model.encode(
        passage_texts, batch_size=256, show_progress_bar=True, normalize_embeddings=True,
    )
    passage_embeddings = np.array(passage_embeddings, dtype=np.float32)
    print(f'  Embedded {len(passage_texts)} passages in {time.time() - t0:.1f}s')

    print('EMBEDDING QUERIES...')
    t0 = time.time()
    query_texts = [ex['question'] for ex in ds]
    query_embeddings = model.encode(
        query_texts, batch_size=256, show_progress_bar=True, normalize_embeddings=True,
    )
    query_embeddings = np.array(query_embeddings, dtype=np.float32)
    print(f'  Embedded {len(query_texts)} queries in {time.time() - t0:.1f}s')

    question_passages = {}
    emb_idx = 0
    for qi, pi, title, text in all_passages:
        if qi not in question_passages:
            question_passages[qi] = {'titles': [], 'embeddings': []}
        question_passages[qi]['titles'].append(title)
        question_passages[qi]['embeddings'].append(passage_embeddings[emb_idx])
        emb_idx += 1

    lsr = LocalSpectralRetrieval(n_components=1, sampling_method='quantile')
    topk = TopKRetrieval()
    mmr = MMRRetrieval()
    methods = {'Top-K': topk, 'MMR': mmr, 'LSR': lsr}

    print()
    print('=' * 70)
    print(f'RUNNING BENCHMARK  (n={NUM_QUESTIONS}, k={K}, threshold={THRESHOLD})')
    print('=' * 70)

    results = {name: {'redundancy': [], 'recall': [], 'variance_coverage': []} for name in methods}
    pc1_ratios = []
    per_query_results = []

    for qi in range(NUM_QUESTIONS):
        qp = question_passages[qi]
        corpus_embs = np.array(qp['embeddings'], dtype=np.float32)
        titles_arr = np.array(qp['titles'])
        q_emb = query_embeddings[qi]
        gold_sf_titles = set(ds[qi]['supporting_facts']['title'])
        pc1 = pc1_variance_ratio(corpus_embs)
        pc1_ratios.append(pc1)

        query_record = {
            'question_idx': qi, 'question': query_texts[qi],
            'n_passages': len(corpus_embs), 'pc1_variance_ratio': pc1,
            'n_gold_titles': len(gold_sf_titles), 'methods': {},
        }

        for name, retriever in methods.items():
            if name == 'Top-K':
                sel_embs, sel_idx = retriever.retrieve(q_emb, corpus_embs, K)
            elif name == 'MMR':
                sel_embs, sel_idx = retriever.retrieve(
                    q_emb, corpus_embs, K, threshold=THRESHOLD, lambda_param=MMR_LAMBDA)
            else:
                sel_embs, sel_idx = retriever.retrieve(
                    q_emb, corpus_embs, K, threshold=THRESHOLD)

            if len(sel_idx) == 0:
                results[name]['redundancy'].append(np.nan)
                results[name]['recall'].append(0.0)
                results[name]['variance_coverage'].append(0.0)
                query_record['methods'][name] = {
                    'n_selected': 0, 'redundancy': None, 'recall': 0.0, 'variance_coverage': 0.0}
                continue

            red = avg_pairwise_cosine(sel_embs)
            rec = supporting_fact_recall(sel_idx, titles_arr, gold_sf_titles)
            vc = variance_coverage(sel_embs, corpus_embs)
            results[name]['redundancy'].append(red)
            results[name]['recall'].append(rec)
            results[name]['variance_coverage'].append(vc)
            query_record['methods'][name] = {
                'n_selected': int(len(sel_idx)), 'redundancy': float(red),
                'recall': float(rec), 'variance_coverage': float(vc),
                'selected_indices': sel_idx.tolist(),
            }
        per_query_results.append(query_record)
        if (qi + 1) % 50 == 0:
            print(f'  Processed {qi + 1}/{NUM_QUESTIONS} questions...')

    print()
    print('=' * 70)
    print('RESULTS SUMMARY')
    print('=' * 70)

    pc1_arr = np.array(pc1_ratios)
    print()
    print('Neighborhood PC1 variance ratio (characterizes collapse):')
    print(f'  Mean:   {np.mean(pc1_arr):.4f}')
    print(f'  Median: {np.median(pc1_arr):.4f}')
    print(f'  Std:    {np.std(pc1_arr):.4f}')
    print(f'  Min:    {np.min(pc1_arr):.4f}')
    print(f'  Max:    {np.max(pc1_arr):.4f}')
    pct50 = (pc1_arr > 0.50).sum()
    pct70 = (pc1_arr > 0.70).sum()
    print(f'  >0.50:  {pct50}/{len(pc1_arr)} ({100*(pc1_arr > 0.50).mean():.1f}%)')
    print(f'  >0.70:  {pct70}/{len(pc1_arr)} ({100*(pc1_arr > 0.70).mean():.1f}%)')

    hdr = f"{'Method':<10} {'Redundancy':>12} {'SF Recall':>12} {'Var Coverage':>14}"
    print()
    print(hdr)
    print('-' * len(hdr))

    summary = {}
    for name in methods:
        red_arr = np.array(results[name]['redundancy'])
        rec_arr = np.array(results[name]['recall'])
        vc_arr = np.array(results[name]['variance_coverage'])
        valid_red = red_arr[~np.isnan(red_arr)]
        mean_red = float(np.mean(valid_red)) if len(valid_red) > 0 else float('nan')
        mean_rec = float(np.mean(rec_arr))
        mean_vc = float(np.mean(vc_arr))
        print(f'{name:<10} {mean_red:>12.4f} {mean_rec:>12.4f} {mean_vc:>14.4f}')
        summary[name] = {
            'redundancy_mean': mean_red,
            'redundancy_std': float(np.std(valid_red)) if len(valid_red) > 0 else 0,
            'recall_mean': mean_rec, 'recall_std': float(np.std(rec_arr)),
            'variance_coverage_mean': mean_vc, 'variance_coverage_std': float(np.std(vc_arr)),
            'n_empty': int(np.isnan(red_arr).sum()),
        }

    print()
    print('  (Redundancy: lower is better | SF Recall & Var Coverage: higher is better)')

    print()
    print('-' * 70)
    print('DELTAS vs Top-K baseline')
    print('-' * 70)
    baseline = summary['Top-K']
    for name in ['MMR', 'LSR']:
        s = summary[name]
        dr = s['redundancy_mean'] - baseline['redundancy_mean']
        drec = s['recall_mean'] - baseline['recall_mean']
        dvc = s['variance_coverage_mean'] - baseline['variance_coverage_mean']
        print(f'  {name}: Redundancy {dr:+.4f}  |  SF Recall {drec:+.4f}  |  Var Coverage {dvc:+.4f}')

    output = {
        'config': {
            'num_questions': NUM_QUESTIONS, 'k': K, 'threshold': THRESHOLD,
            'mmr_lambda': MMR_LAMBDA, 'model': MODEL_NAME,
        },
        'pc1_distribution': {
            'mean': float(np.mean(pc1_arr)), 'median': float(np.median(pc1_arr)),
            'std': float(np.std(pc1_arr)), 'min': float(np.min(pc1_arr)),
            'max': float(np.max(pc1_arr)), 'values': pc1_ratios,
        },
        'summary': summary,
        'per_query': per_query_results,
    }

    with open(RESULTS_PATH, 'w') as f:
        json.dump(output, f, indent=2)
    print(''); print(f'Results saved to {RESULTS_PATH}')
    print('Done.')


if __name__ == '__main__':
    main()
