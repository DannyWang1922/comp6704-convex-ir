"""Training-only descriptive analysis of frozen parameters; no qrels file reads.

Reconstruct binary relevance from the full training triples. This establishes
positive-set recall and binary nDCG. BEIR Table 1 reports SciFact as Binary
(https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/file/65b9eea6e1cc6bb9f0cd2a47751a186f-Paper-round2.pdf).
Local relevance grades are not opened or individually checked.
No optimizer, validation evaluator, or test evaluator is imported or invoked.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import time
import numpy as np
from threadpoolctl import threadpool_limits
from query_adapt.objective import PairwiseObjective
from query_adapt.retrieval import retrieve

SEEDS=(42,43,44)
LAMBDAS=(.01,.001,.0001,.00001)
BINARY_EVIDENCE={"source":"BEIR original paper, Table 1, SciFact: Binary", "url":"https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/file/65b9eea6e1cc6bb9f0cd2a47751a186f-Paper-round2.pdf", "local_grade_values_read":False}


class AllowedInputs:
    def __init__(self, paths):
        self.allowed={Path(path).resolve() for path in paths}
        self.read_paths=set()
    def check(self,path):
        path=Path(path).resolve()
        if path not in self.allowed:
            raise ValueError(f'input is outside the explicit training-only allowlist: {path}')
        self.read_paths.add(path)
        return path
    def json(self,path):
        return json.loads(self.check(path).read_text())
    def npz(self,path):
        with np.load(self.check(path),allow_pickle=False) as values:
            return {key:values[key] for key in values.files}


def reconstruct_binary_qrels(rows, train_ids, pair_config):
    expected=set(train_ids)
    qrels={qid:{} for qid in sorted(expected)}
    grouped={}
    for row in rows:
        qid=row['query_id'];positive=row['positive_id'];negative=row['negative_id']
        if qid not in expected or row['kind'] not in {'hard','random'}:
            raise ValueError('non-training query or unknown pair kind')
        if positive==negative:
            raise ValueError('positive equals negative')
        qrels[qid][positive]=1
        grouped.setdefault((qid,positive),[]).append(row)
    if any(not positives for positives in qrels.values()):
        raise ValueError('training positive coverage is incomplete')
    for group in grouped.values():
        counts=Counter(row['kind'] for row in group)
        if counts != {'hard':pair_config['hard_negatives'],'random':pair_config['random_negatives']}:
            raise ValueError('full pair multiplicity does not match preparation settings')
        if len({row['negative_id'] for row in group})!=len(group):
            raise ValueError('duplicate negatives for a training positive')
        if any(row['negative_id'] in qrels[row['query_id']] for row in group):
            raise ValueError('known positive appears as a sampled negative')
    return qrels


def binary_metrics(qrels, rankings):
    records=[]
    discounts=1/np.log2(np.arange(2,12))
    if set(qrels)!=set(rankings):raise ValueError('ranking query set differs')
    for qid in sorted(qrels):
        relevant=set(qrels[qid]);ranking=list(rankings[qid])
        gains=np.array([doc in relevant for doc in ranking[:10]],dtype=float)
        ideal=float(discounts[:min(10,len(relevant))].sum())
        ndcg=float(gains @ discounts[:len(gains)]/ideal)
        hits=len(relevant.intersection(ranking[:100]))
        records.append({'query_id':qid,'nDCG@10':ndcg,'Recall@100':hits/len(relevant),
                        'relevant_count':len(relevant),'relevant_hits_at_100':hits,'ranking_top100':ranking[:100]})
    return {'nDCG@10':float(np.mean([r['nDCG@10'] for r in records])),
            'Recall@100':float(np.mean([r['Recall@100'] for r in records]))},records


def write_new(path,value):
    text=json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n'
    with Path(path).open('x') as stream:stream.write(text)


def analyze(root, output):
    root=Path(root).resolve();output=Path(output).resolve()
    analysis_home=(root/'artifacts/analysis').resolve()
    runs_home=(root/'artifacts/runs').resolve()
    if output==analysis_home or not output.is_relative_to(analysis_home) or output.is_relative_to(runs_home):
        raise ValueError('output must be a new child directory of root/artifacts/analysis, never runs or an ancestor')
    if output.exists():raise FileExistsError(output)
    campaign=root/'artifacts/runs/full-experiment-001'
    diagnostics=root/'artifacts/runs/adam-rates/diagnostics'
    embeddings=root/'artifacts/embeddings.npz';splits=root/'artifacts/data/splits.json'
    cert_paths={(seed,lam):campaign/'targets'/f'seed{seed}-lambda{lam:g}-n5936'/'certified-newton.json' for seed in SEEDS for lam in LAMBDAS}
    source_paths=[root/'src/query_adapt/data.py',root/'src/query_adapt/retrieval.py',Path(__file__).resolve()]
    paths=[embeddings,splits,campaign/'manifest.json',campaign/'e1-summary.json',*source_paths,*cert_paths.values()]
    paths += [diagnostics/f'seed{seed}.{suffix}' for seed in SEEDS for suffix in ('npz','json')]
    inputs=AllowedInputs(paths)
    # Only the internal training ID list is used. No label file is opened.
    train_ids=inputs.json(splits)['train']
    if len(train_ids)!=647 or len(set(train_ids))!=647:raise ValueError('expected frozen 647 training queries')
    e1=inputs.json(campaign/'e1-summary.json')
    if e1['state']!='complete' or e1['mode']!='formal' or len(e1['targets'])!=12:
        raise ValueError('expected completed frozen E1')
    manifest=inputs.json(campaign/'manifest.json')
    arrays=inputs.npz(embeddings)
    docs,queries=arrays['documents'],arrays['queries']
    dids,qids=arrays['doc_ids'].tolist(),arrays['query_ids'].tolist()
    if len(dids)!=len(docs) or len(qids)!=len(queries) or len(set(dids))!=len(dids) or len(set(qids))!=len(qids):
        raise ValueError('embedding ID alignment failed')
    if not set(train_ids)<=set(qids) or docs.shape!=(5183,384) or queries.shape[1]!=384:
        raise ValueError('embedding shape or training query coverage differs')
    if not np.isfinite(docs).all() or not np.isfinite(queries).all():raise ValueError('nonfinite embeddings')
    datasets={};qrels=None;hard_rows=None;hard_x=None
    for seed in SEEDS:
        metadata=inputs.json(diagnostics/f'seed{seed}.json')
        x=inputs.npz(diagnostics/f'seed{seed}.npz')['features']
        rows=metadata['triples']
        if metadata['seed']!=seed or x.shape!=(5936,384) or len(rows)!=5936 or not np.isfinite(x).all():
            raise ValueError('training feature identity differs')
        rebuilt=reconstruct_binary_qrels(rows,train_ids,metadata['pairs'])
        if not set(doc for relevant in rebuilt.values() for doc in relevant)<=set(dids):
            raise ValueError('training positive document missing from embeddings')
        mask=np.array([row['kind']=='hard' for row in rows])
        selected_rows=[row for row in rows if row['kind']=='hard']
        if qrels is None:qrels=rebuilt;hard_rows=selected_rows;hard_x=x[mask]
        elif qrels!=rebuilt or hard_rows!=selected_rows or not np.array_equal(hard_x,x[mask]):
            raise ValueError('seed training positives or deterministic hard pairs differ')
        datasets[seed]=(x,rows,mask)
    if sum(len(relevant) for relevant in qrels.values())!=742:raise ValueError('expected all 742 training positives')
    certified={}
    for (seed,lam),path in cert_paths.items():
        cert=inputs.json(path)
        if cert['method']!='newton_refined' or cert['verified'] is not True or (cert['seed'],cert['regularization'],cert['n'])!=(seed,lam,5936):
            raise ValueError('certificate identity differs')
        source=cert['source']
        if Path(source['training_cache']).resolve()!=diagnostics/f'seed{seed}.npz' or Path(source['training_metadata']).resolve()!=diagnostics/f'seed{seed}.json' or Path(source['run_manifest']).resolve()!=campaign/'manifest.json':
            raise ValueError('certificate references data outside the frozen allowlist')
        a=np.asarray(cert['parameters'],dtype=float)
        obj=PairwiseObjective(datasets[seed][0],lam)
        _,gradient=obj.value_gradient(a)
        norm=float(np.linalg.norm(gradient))
        if norm**2/(2*lam)>1e-12 or norm/lam>1e-5:raise ValueError('frozen parameter no longer satisfies strict certificate')
        certified[seed,lam]=a
    for source in source_paths:inputs.check(source) # Whitelist source snapshots too.
    output.mkdir(parents=True,exist_ok=False);(output/'per-query').mkdir();(output/'source').mkdir()
    for source in source_paths:shutil.copy2(source,output/'source'/source.name)
    provenance={'scope':'posthoc_training_only','campaign_commit':manifest['git_commit'],'allowlisted_inputs':[str(p) for p in sorted(inputs.allowed)],
        'inputs_read':[str(p) for p in sorted(inputs.read_paths)],'training_query_ids':sorted(train_ids),'training_positive_pairs':742,
        'qrels_definition':'binary relevance reconstructed from every positive_id in complete training triples, consistent with BEIR Table 1 reporting SciFact Binary; no local qrels file or grade value was read, and local byte-for-byte equivalence was not checked',
        'binary_relevance_evidence':BINARY_EVIDENCE,
        'heldout_qrels_read':False,'optimizer_run':False,'hyperparameters_changed':False,
        'coverage_checks':'all 647 train IDs, 742 positives, 5936 rows per seed, expected pair multiplicity; positives and hard features identical across seeds'}
    write_new(output/'provenance.json',provenance)
    write_new(output/'reconstructed-training-binary-qrels.json',qrels)
    start=time.perf_counter()
    def measure(label,a,seed=None,lam=None):
        tick=time.perf_counter()
        rankings=retrieve(docs,queries,dids,qids,sorted(train_ids),a,top_k=100)
        metrics,perquery=binary_metrics(qrels,rankings)
        perquery_file=f'per-query/{label}.json'
        write_new(output/perquery_file,{'split':'training','relevance':'reconstructed_binary','query_count':len(perquery),'records':perquery})
        hard=hard_x if seed is None else datasets[seed][0][datasets[seed][2]]
        return {'seed':seed,'regularization':lam,'parameter_distance':float(np.linalg.norm(a-1)),
            'parameter_norm':float(np.linalg.norm(a)),'negative_dimensions':int(np.sum(a<0)),
            'hard_pair_accuracy':float(np.mean(hard@a>0)),'hard_pair_count':len(hard),
            'train_metrics':metrics,'per_query_file':perquery_file,'seconds':time.perf_counter()-tick,
            'certificate_source':str(cert_paths[seed,lam]) if seed is not None else None}
    with threadpool_limits(limits=1):
        baseline=measure('baseline',np.ones(384))
        systems=[]
        for seed in SEEDS:
            for lam in LAMBDAS:
                result=measure(f'seed{seed}-lambda{lam:g}',certified[seed,lam],seed,lam)
                systems.append(result)
                print(f'TRAIN_POSTHOC seed={seed} lambda={lam:g} distance={result["parameter_distance"]:.6f} negative={result["negative_dimensions"]} hard_accuracy={result["hard_pair_accuracy"]:.6f} binary_ndcg10={result["train_metrics"]["nDCG@10"]:.6f}',flush=True)
    summary={'scope':'posthoc_training_only','analysis':'descriptive frozen-parameter analysis, not preregistered or causal',
             'qrels_definition':provenance['qrels_definition'],'baseline':baseline,'systems':systems,
             'query_count':647,'training_positive_pairs':742,'seconds':time.perf_counter()-start,
             'heldout_qrels_read':False,'optimizer_run':False}
    write_new(output/'summary.json',summary)
    print(f'TRAIN_POSTHOC COMPLETE systems={1+len(systems)} output={output}',flush=True)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output',type=Path,default=Path('artifacts/analysis/posthoc-001'))
    args=parser.parse_args()
    analyze(args.root,args.output)

if __name__=='__main__':main()
