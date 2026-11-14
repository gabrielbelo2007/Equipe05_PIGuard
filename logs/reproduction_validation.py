"""Five evaluation blocks for the seed-42 / 512-token reproduction.

No API requests are made. All metrics are percentages; missing values stay missing.
"""
import csv
import gc
import hashlib
import json
import platform
import time
from pathlib import Path

BLOCKS = ['Tabelas_1_e_7_Baselines_PIGuard', 'Tabela_2_Designs_1_e_2_Fase1_Triggers',
          'Tabela_2_Ablacao_Completa', 'Tabela_3_Shortcut_Mitigation', 'Extra_TensorTrust']
BASES = ['train_cleaned_original', 'train_augmented_original', 'train_augmented_rp']
VARIANTS = ['original', 'semi_rp', 'rp']
LOCAL_MODELS = BASES + [f'train_{prefix}_{v}' for v in VARIANTS for prefix in
    ['clean_mof', 'augmented_mof_finetuning', 'augmented_mof']] + [f'train_augmented_shortcut_{v}' for v in ['original', 'rp']]
HF_MODELS = {'Fmops': 'fmops/distilbert-prompt-injection', 'Deepset': 'deepset/deberta-v3-base-injection',
    'PromptGuard': 'meta-llama/Prompt-Guard-86M', 'ProtectAIv2': 'ProtectAI/deberta-v3-base-prompt-injection-v2', 'PIGuard HF': 'leolee99/PIGuard'}
METRICS = ['overdefense', 'benign', 'malicious', 'average']
PUBLIC_SETS = ['NotInject_one', 'NotInject_two', 'NotInject_three', 'WildGuard', 'BIPIA_text', 'BIPIA_code']
TABLE7 = {'Deepset': [23.89,52.21,12.39,50.98,84.67], 'Fmops':[18.58,53.10,13.27,50.88,88.67],
 'PromptGuard':[.88,0,0,6.69,100], 'ProtectAIv2':[77.88,47.79,46.02,75.18,8.67],
 'PIGuard':[91.15,89.38,81.42,76.11,68.34]}
DESIGNS = [('Basic', [75.22,78.53,70.17,74.64]), ('+ Aug',[64.31,81.36,75.95,73.87]),
 ('Basic + MOF scratch',[89.38,84.73,71.57,81.89]), ('+ Aug + MOF fine-tuning',[68.14,82.11,74.81,75.02]),
 ('+ Aug + MOF scratch',[87.32,85.74,77.39,83.48])]


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)); tmp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()


def csv_write(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row)) or ['status']
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fields); writer.writeheader(); writer.writerows(rows)


def aggregate(scores):
    over = sum(scores[k] for k in PUBLIC_SETS[:3]) / 3
    malicious = (scores['BIPIA_text'] + scores['BIPIA_code']) / 2
    return dict(overdefense=over, benign=scores['WildGuard'], malicious=malicious,
                average=(over + scores['WildGuard'] + malicious) / 3)


def model_for_design(index, variant):
    if index == 0: return BASES[0]
    if index == 1: return BASES[2] if variant == 'rp' else BASES[1]
    return f'train_{["", "", "clean_mof", "augmented_mof_finetuning", "augmented_mof"][index]}_{variant}'


class ValidationRun:
    def __init__(self, repo, output, legacy=None, batch_size=32, force=False, factory=None):
        self.repo, self.output = Path(repo).resolve(), Path(output).resolve()
        self.root = self.output / 'Gráficos'
        self.legacy = Path(legacy) if legacy else None
        self.batch_size, self.force, self.factory = batch_size, force, factory
        import torch, transformers
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.environment = {'torch': torch.__version__, 'transformers': transformers.__version__,
                            'python': platform.python_version(), 'device': self.device}
        self.data = {}; self.hashes = {}
        for name in PUBLIC_SETS + ['tensor_trust_hijacking', 'tensor_trust_extraction']:
            file = 'wildguard' if name == 'WildGuard' else name
            path = self.repo / 'datasets' / (file + '.json')
            data = read(path)
            texts = [x['prompt'] for x in data] if isinstance(data, list) else [x for group in data.values() for x in group]
            if not texts: raise ValueError(f'Benchmark vazio: {path}')
            self.data[name] = texts; self.hashes[name] = digest(path)
        self.results = {}; self.status = {}
        for folder in BLOCKS: (self.root / folder).mkdir(parents=True, exist_ok=True)

    def evaluator(self, name):
        if self.factory: return self.factory(name)
        from Eval.validate_models_table1 import HFSequenceClassifierEvaluator, LocalPIGuardEvaluator
        if name in HF_MODELS: return HFSequenceClassifierEvaluator(HF_MODELS[name], self.device)
        return LocalPIGuardEvaluator(str(self.output / name / 'best_model.pth'), self.device)

    def fingerprint(self, name, evaluator):
        config = getattr(evaluator.model, 'config', getattr(evaluator.model, 'deberta', None))
        revision = getattr(config, '_commit_hash', None)
        if revision is None and hasattr(config, 'config'): revision = getattr(config.config, '_commit_hash', None)
        ckpt = self.output / name / 'best_model.pth'
        return {'seed':42, 'max_length':512, 'batch_size':self.batch_size, 'data':self.hashes,
                'checkpoint':digest(ckpt) if name not in HF_MODELS else HF_MODELS[name],
                'revision':revision, 'environment':self.environment, 'implementation':digest(__file__),
                'evaluator_implementation':digest(self.repo / 'Eval/validate_models_table1.py')}

    def evaluate(self, name):
        import torch
        self.results.pop(name, None)
        if name not in HF_MODELS and not (self.output / name / 'best_model.pth').is_file():
            self.status[name] = 'checkpoint ausente'; self.persist(); return
        evaluator = None
        try:
            evaluator = self.evaluator(name); evaluator.model.eval()
            spec = self.fingerprint(name, evaluator)
            path = self.root / '_cache' / (name + '.json')
            cached = read(path) if path.is_file() else {}
            if not self.force and cached.get('spec') == spec and cached.get('status') == 'complete':
                self.results[name] = cached; self.status[name] = 'cache compatível'; return
            scores = {}; counts = {}; elapsed = 0.; total = 0
            with torch.inference_mode():
                for key, texts in self.data.items():
                    correct = 0; target = 0 if key in PUBLIC_SETS[:4] else 1
                    for i in range(0,len(texts),self.batch_size):
                        batch = texts[i:i+self.batch_size]
                        if self.device == 'cuda': torch.cuda.synchronize()
                        start = time.perf_counter()
                        preds = evaluator.predict_batch(batch, max_length=512)
                        if self.device == 'cuda': torch.cuda.synchronize()
                        elapsed += time.perf_counter() - start; total += len(batch)
                        if len(preds) != len(batch): raise ValueError('Quantidade incorreta de predições')
                        correct += sum(p == target for p in preds)
                    scores[key] = correct / len(texts) * 100; counts[key] = {'correct':correct, 'n':len(texts)}
            payload = dict(spec=spec,status='complete',source='nova avaliação',scores=scores,counts=counts,
                           metrics=aggregate(scores),inference_ms=elapsed/total*1000)
            payload['metrics']['inference_ms'] = payload['inference_ms']
            payload['metrics']['efficiency'] = payload['metrics']['average'] / payload['inference_ms'] if elapsed else None
            write(path,payload); self.results[name] = payload; self.status[name] = 'concluído'
        except Exception as exc:
            self.status[name] = f'erro: {type(exc).__name__}: {exc}'
        finally:
            del evaluator; gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            self.persist()

    def persist(self):
        write(self.root / 'manifesto_avaliacao.json', {'max_length':512,'seed':42,'environment':self.environment,
             'datasets':self.hashes,'status':self.status,'pint_included':False})

    def historical(self):
        """Read paid/remote baseline records only; never call their endpoints."""
        result = {}
        candidates = [] if self.legacy is None else [self.legacy / 'Gráficos/table1_validation_raw.json', self.legacy / 'table1_validation_raw.json']
        for path in candidates:
            if not path.is_file(): continue
            raw = read(path)
            for key,value in raw.items():
                if isinstance(value,dict) and any(n in key.lower() for n in ['gpt','llama','lakera']):
                    result[key] = {'source':'histórico importado', 'origin':str(path), 'max_length':'histórico; não confirmado', 'record':value}
            break
        if not result:
            existing = self.root / BLOCKS[0] / 'table1_validation_raw.json'
            if existing.is_file():
                saved = read(existing).get('historical', {})
                if isinstance(saved, dict):
                    result = saved
        return result

    def load_cache(self):
        """Load cached evaluation results and manifest status without running inference."""
        manifest_path = self.root / 'manifesto_avaliacao.json'
        if manifest_path.is_file():
            manifest = read(manifest_path)
            if isinstance(manifest.get('status'), dict):
                self.status.update(manifest['status'])
        for name in list(HF_MODELS) + LOCAL_MODELS:
            cache_path = self.root / '_cache' / f'{name}.json'
            if cache_path.is_file():
                cached = read(cache_path)
                if cached.get('status') == 'complete':
                    self.results[name] = cached
                    self.status.setdefault(name, 'cache carregado')
        return self.results

    def rows(self, names):
        return [dict(model=n, source='nova avaliação', max_length=512, status=self.status.get(n,'pendente'),
                     **self.results.get(n,{}).get('metrics',{})) for n in names]

    def _controlled_refs(self):
        refs = {}
        for ref_name in ['PIGuard HF', 'train_augmented_mof_original']:
            metrics = self.results.get(ref_name, {}).get('metrics')
            if isinstance(metrics, dict) and 'average' in metrics:
                refs[ref_name] = metrics
        return refs

    def _add_ref_deltas(self, row, ref_map, keys=METRICS, prefix_map=None):
        prefix_map = prefix_map or {'PIGuard HF': 'hf', 'train_augmented_mof_original': 'mof_orig'}
        out = dict(row)
        if 'average' not in out:
            return out
        for ref_name, ref_metrics in ref_map.items():
            if not ref_metrics or 'average' not in ref_metrics:
                continue
            tag = prefix_map.get(ref_name, ref_name)
            for k in keys:
                if isinstance(out.get(k), (int, float)) and isinstance(ref_metrics.get(k), (int, float)):
                    out[f'delta_{tag}_{k}_pp'] = out[k] - ref_metrics[k]
        return out

    def plot(self, path, rows, metrics=METRICS, title='', zero=False):
        import matplotlib.pyplot as plt
        import numpy as np
        rows = [r for r in rows if any(isinstance(r.get(k),(int,float)) for k in metrics)]
        if not rows: return
        fig,ax = plt.subplots(figsize=(max(9,len(rows)*.9),5),layout='constrained')
        x=np.arange(len(rows)); width=.8/len(metrics)
        for j,k in enumerate(metrics):
            vals=[r[k] if isinstance(r.get(k),(int,float)) else float('nan') for r in rows]
            ax.bar(x+(j-(len(metrics)-1)/2)*width,vals,width,label=k)
        ax.set_xticks(x,[r.get('model',r.get('design','')) for r in rows],rotation=65,ha='right')
        ax.set_ylabel('Pontos percentuais' if zero else 'Acurácia (%)'); ax.set_title(title)
        if zero: ax.axhline(0,color='black',linewidth=.7)
        else: ax.set_ylim(0,105)
        ax.legend(); path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
        fig.savefig(path,dpi=150,bbox_inches='tight'); plt.close(fig)

    def plot_ablation_deltas(self, path, delta_rows, refs):
        import matplotlib.pyplot as plt
        import numpy as np
        ref_order = [r for r in ['train_augmented_mof_original', 'PIGuard HF'] if r in refs]
        panels = [(r, [row for row in delta_rows if row.get('reference') == r]) for r in ref_order]
        panels = [(r, rows) for r, rows in panels if rows]
        if not panels:
            return
        fig, axes = plt.subplots(len(panels), 1, figsize=(15, 4.8 * len(panels)), squeeze=False, layout='constrained')
        titles = {
            'train_augmented_mof_original': 'Deltas frente a train_augmented_mof_original (controle interno sem PINT)',
            'PIGuard HF': 'Deltas frente a PIGuard HF (modelo oficial publicado, sem PINT)',
        }
        for ax, (ref_name, subrows) in zip(axes.flat, panels):
            x = np.arange(len(subrows))
            width = 0.8 / len(METRICS)
            for j, k in enumerate(METRICS):
                vals = [r[k] if isinstance(r.get(k), (int, float)) else float('nan') for r in subrows]
                ax.bar(x + (j - (len(METRICS) - 1) / 2) * width, vals, width, label=k)
            labels = [r.get('label', r.get('model', '')) for r in subrows]
            ax.set_xticks(x, labels, rotation=45, ha='right', fontsize=8.5)
            ax.axhline(0, color='black', linewidth=0.8)
            ax.grid(axis='y', alpha=0.25, linestyle='--')
            ax.set_ylabel('Pontos percentuais (pp)')
            ax.set_title(titles.get(ref_name, f'Deltas frente a {ref_name}'))
            ax.legend(loc='lower left', ncol=4)
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches='tight'); plt.close(fig)

    def panel_plot(self,path,rows):
        import matplotlib.pyplot as plt
        rows=[r for r in rows if any(isinstance(r.get(k),(float,int)) for k in METRICS)]
        if not rows:return
        cols=3; count=(len(rows)+cols-1)//cols
        fig,axes=plt.subplots(count,cols,figsize=(14,count*3.5),squeeze=False,layout='constrained')
        for ax,row in zip(axes.flat,rows):
            ax.bar(METRICS,[row.get(k,float('nan')) for k in METRICS]);ax.set_ylim(0,105)
            ax.set_title(row['model']+' / '+row.get('source',''),fontsize=8);ax.tick_params(axis='x',rotation=30)
        for ax in list(axes.flat)[len(rows):]:ax.set_visible(False)
        fig.savefig(path,dpi=130,bbox_inches='tight');plt.close(fig)

    def reports(self):
        from Eval.validate_models_table1 import PAPER_TABLE_1
        import matplotlib.pyplot as plt
        import numpy as np
        b1,b2,b3,b4,b5=[self.root / x for x in BLOCKS]
        refs = self._controlled_refs()
        finals=[f'train_augmented_mof_{v}' for v in VARIANTS]
        rows=self.rows(list(HF_MODELS)+finals)
        history=self.historical()
        for paid in ['GPT-4o', 'Llama-2-chat', 'LlamaGuard3', 'LakeraGuard']:
            if not any(paid.lower() in key.lower() for key in history):
                self.status[paid] = 'histórico ausente; nenhuma chamada de API'
                rows.append(dict(model=paid,source='histórico indisponível',status='pendente'))
        for n,h in history.items(): rows.append(dict(model=n,source=h['source'],max_length=h['max_length'],**{k:v for k,v in h['record'].items() if k in METRICS}))
        write(b1/'table1_validation_raw.json', {
            'evaluated':{n:self.results[n] for n in list(HF_MODELS)+finals if n in self.results},
            'historical':history,
            'controlled_references_no_pint':refs,
            'paper_reference':PAPER_TABLE_1,
        })
        csv_write(b1/'table1_validation_summary.csv',rows)
        deltas=[]; subrows=[]
        hist_notes = {
            'GPT-4o': 'Histórico real sem PINT (openai/gpt-4o)',
            'Llama-2-chat': 'Histórico sem PINT (substituto llama-3.1-8b-instruct)',
            'LlamaGuard3': 'Histórico sem PINT (substituto llama-guard-4-12b)',
        }
        for row in rows:
            n=row['model']
            src = row.get('source', '')
            if 'average' in row:
                if src == 'nova avaliação':
                    comp = 'Mesma composição sem PINT (512 tokens)'
                elif src == 'histórico importado' and 'lakeraguard' not in n.lower():
                    comp = next((v for k, v in hist_notes.items() if k.lower() in n.lower()), 'Histórico sem PINT')
                else:
                    comp = None
                if comp:
                    for ref_name, ref_metrics in refs.items():
                        deltas.append(dict(model=n, source=src, reference=ref_name,
                                           **{k: row[k] - ref_metrics[k] for k in METRICS}, comparison=comp))
            ref7=TABLE7.get(n,TABLE7['PIGuard'] if n in finals or n=='PIGuard HF' else None)
            if ref7 and n in self.results:
                m=self.results[n]
                for key,paper in zip(PUBLIC_SETS[:4]+['BIPIA'],ref7):
                    actual=m['metrics']['malicious'] if key=='BIPIA' else m['scores'][key]
                    subrows.append(dict(model=n,dataset=key,paper=paper,reproduced=actual,delta_pp=actual-paper))
        csv_write(b1/'table1_detailed_deltas.csv',deltas);csv_write(b1/'table7_submetrics_comparison.csv',subrows)
        self.plot(b1/'table1_comparison.png',rows,title='Tabela 1: nova avaliação sem PINT; históricos identificados no CSV')
        self.panel_plot(b1/'table1_all_models_comparison_grid.png', rows)
        for i,row in enumerate(rows):self.plot(b1/'individual_charts'/f'compare_{i+1}.png',[row],title=row['model']+' — '+row['source'])
        base_rows=[self._add_ref_deltas(r, refs) for r in self.rows(BASES)]
        write(b2/'evaluation_comparison_raw.json',{n:self.results[n] for n in BASES if n in self.results})
        csv_write(b2/'evaluation_comparison_table.csv',base_rows);self.plot(b2/'benchmark_comparison.png',base_rows,title='Fase 1 — benchmarks públicos (sem PINT)')
        ablation=[]; ablation_deltas=[]
        for i,(design,_) in enumerate(DESIGNS):
            for v in VARIANTS:
                n=model_for_design(i,v);row=dict(design=design,variant=v,**self.rows([n])[0]);ablation.append(row)
                if n in self.results:
                    for ref_name, ref_metrics in refs.items():
                        ablation_deltas.append(dict(
                            design=design, variant=v, model=n, reference=ref_name,
                            label=f'D{i+1} {design} ({v})',
                            **{k: row[k] - ref_metrics[k] for k in METRICS},
                            comparison=f'Mesma composição sem PINT vs. {ref_name}',
                        ))
            self.plot(b3/'individual_charts'/f'design_{i+1}.png',[dict(r,model=r['variant']) for r in ablation if r['design']==design],title=design)
        write(b3/'ablation_results_raw.json',{
            'controlled_references_no_pint': refs,
            'paper_table_2': DESIGNS,
            'evaluated_experiments': {n: self.results[n] for n in LOCAL_MODELS if n in self.results},
        })
        csv_write(b3/'ablation_table_2_summary.csv',ablation)
        csv_write(b3/'ablation_table_2_detailed_deltas.csv',
                  [{k: v for k, v in r.items() if k != 'label'} for r in ablation_deltas])
        self.panel_plot(b3/'ablation_mosaic_all_designs.png', [dict(r,model=r['design']+' / '+r['variant']) for r in ablation])
        fig,axes=plt.subplots(1,4,figsize=(18,5),layout='constrained')
        short_designs = ['D1\nBasic', 'D2\n+Aug', 'D3\nBasic\n+MOF', 'D4\n+Aug\n+FT', 'D5\n+Aug\n+MOF']
        for ax,key in zip(axes,METRICS):
            for variant in VARIANTS:
                selected=[r for r in ablation if r['variant']==variant]
                ax.plot(range(1,6),[r.get(key,float('nan')) for r in selected],marker='o',label=variant)
            if 'PIGuard HF' in refs:
                ax.axhline(refs['PIGuard HF'][key], color='black', linestyle='--', linewidth=1.1, label='PIGuard HF')
            if 'train_augmented_mof_original' in refs:
                ax.axhline(refs['train_augmented_mof_original'][key], color='gray', linestyle=':', linewidth=1.2, label='mof_original')
            ax.set(title=key,xlabel='Desenho da Tabela 2',ylabel='Acurácia (%)',ylim=(0,105))
            ax.set_xticks(range(1,6), short_designs, fontsize=8.5)
            ax.grid(axis='y', alpha=0.25, linestyle='--')
            ax.legend(fontsize=8)
        fig.savefig(b3/'ablation_trajectories_comparison.png',dpi=150,bbox_inches='tight');plt.close(fig)
        self.plot_ablation_deltas(b3/'ablation_discrepancy_deltas.png', ablation_deltas, refs)
        wo_mof_map = {v: self.results.get(f'train_augmented_{v}', {}).get('metrics') for v in ['original', 'rp']}
        def _with_t3_deltas(row):
            v = 'rp' if row['model'].endswith('_rp') else 'original'
            t3_refs = dict(refs)
            if wo_mof_map.get(v):
                t3_refs[f'w/o MOF ({v})'] = wo_mof_map[v]
            pmap = {'PIGuard HF': 'hf', 'train_augmented_mof_original': 'mof_orig', f'w/o MOF ({v})': 'wo_mof'}
            return self._add_ref_deltas(row, t3_refs, keys=['overdefense', 'malicious', 'average'], prefix_map=pmap)
        shortcut=[_with_t3_deltas(r) for r in self.rows([f'train_augmented_shortcut_{v}' for v in ['original','rp']])]
        comparison=[_with_t3_deltas(r) for r in self.rows([f'train_{p}_{v}' for v in ['original','rp'] for p in ['augmented','augmented_shortcut','augmented_mof']])]
        write(b4/'shortcut_mitigation_results.json',{
            'controlled_references_no_pint': {
                **refs,
                **{f'train_augmented_{v}': wo_mof_map[v] for v in ['original', 'rp'] if wo_mof_map.get(v)},
            },
            'paper_reference':{'overdefense':86.73,'malicious':65.53},
            'evaluated_models':{r['model']:self.results.get(r['model']) for r in shortcut},
        })
        csv_write(b4/'shortcut_mitigation_table.csv',shortcut);csv_write(b4/'shortcut_vs_mof_ablation.csv',comparison)
        t3_rows = [r for r in comparison if any(isinstance(r.get(k), (int, float)) for k in ['overdefense', 'malicious'])]
        if t3_rows:
            fig, ax = plt.subplots(figsize=(max(9, len(t3_rows) * 1.1), 5), layout='constrained')
            x = np.arange(len(t3_rows)); width = 0.8 / 2
            colors = ['tab:blue', 'tab:orange']
            for j, (k, c) in enumerate(zip(['overdefense', 'malicious'], colors)):
                vals = [r[k] if isinstance(r.get(k), (int, float)) else float('nan') for r in t3_rows]
                ax.bar(x + (j - 0.5) * width, vals, width, label=k, color=c)
                if 'PIGuard HF' in refs:
                    ax.axhline(refs['PIGuard HF'][k], color=c, linestyle='--', linewidth=1.1, label=f'PIGuard HF ({k})')
            ax.set_xticks(x, [r['model'] for r in t3_rows], rotation=45, ha='right')
            ax.set_ylabel('Acurácia (%)'); ax.set_ylim(0, 105)
            ax.set_title('Tabela 3 — benchmarks públicos (sem PINT; tracejado = PIGuard HF)')
            ax.grid(axis='y', alpha=0.25, linestyle='--')
            ax.legend(ncol=2, fontsize=8.5)
            fig.savefig(b4/'shortcut_vs_mof_comparison.png', dpi=150, bbox_inches='tight'); plt.close(fig)
        tt=[]
        hf=self.results.get('PIGuard HF',{}).get('scores',{})
        hf_avg=sum(hf.get(k,0) for k in ['tensor_trust_hijacking','tensor_trust_extraction'])/2 if hf else None
        orig_tt=self.results.get('train_augmented_mof_original',{}).get('scores',{})
        orig_tt_avg=sum(orig_tt.get(k,0) for k in ['tensor_trust_hijacking','tensor_trust_extraction'])/2 if orig_tt else None
        for n in ['PIGuard HF']+finals:
            scores=self.results.get(n,{}).get('scores',{})
            if scores:
                h,e=[scores[k] for k in ['tensor_trust_hijacking','tensor_trust_extraction']]
                tt.append(dict(
                    model=n,hijacking=h,extraction=e,average=(h+e)/2,
                    delta_hf_pp=(h+e)/2-hf_avg if hf_avg is not None else None,
                    delta_mof_orig_pp=(h+e)/2-orig_tt_avg if orig_tt_avg is not None else None,
                ))
        write(b5/'tensor_trust_results.json',tt);csv_write(b5/'tensor_trust_comparison.csv',tt)
        self.plot(b5/'tensor_trust_comparison.png',tt,['hijacking','extraction','average'],title='TensorTrust — benchmark extra; média macro')
        self.index()

    def index(self):
        text='# Índice da rodada seed 42 / 512 tokens\n\n'
        references=['Tabelas 1 e 7; Figuras 1 e 5','Tabela 2, desenhos 1 e 2; seções 4.1–4.2','Tabela 2 completa; seção 5.3','Tabela 3; seção 5.4','Extra, sem tabela correspondente no artigo']
        for folder,ref in zip(BLOCKS,references):text+=f'- [{folder}]({folder}/): {ref}.\n'
        text+='\nResultados novos usam eval(), seed 42 e 512 tokens. Benign=WildGuard; malicious=média BIPIA text/code. PINT ausente. Os agregados do artigo incluem PINT e não são diretamente comparáveis nas Tabelas 1, 2 e 3; por isso, os deltas dessas tabelas usam como referenciais controlados (sem PINT) o modelo oficial `PIGuard HF` e `train_augmented_mof_original` (enquanto a Tabela 7 compara cada dataset individual diretamente com o artigo). Históricos pagos são importados, com protocolo anterior/desconhecido. GFLOPs não foram medidos nesta rodada.\n\n'
        for name,status in self.status.items(): text+=f'- {name}: {status}\n'
        (self.root/'INDICE.md').write_text(text)

    def recheck(self):
        import subprocess,sys
        for name in BASES:
            if name not in self.results: continue
            folder=self.root/BLOCKS[1]; path=folder/f'trigger_words_{name}.json'
            spec=self.results[name]['spec'];meta=path.with_suffix('.manifest.json')
            if not self.force and path.exists() and meta.exists() and read(meta)==spec:continue
            if meta.exists(): meta.unlink()
            if path.exists(): path.unlink()
            cmd=[sys.executable,str(self.repo/'datasets/MOF/Cache/token_recheck.py'),'--checkpoint',str(self.output/name/'best_model.pth'),'--output',str(path),'--batch_size','512']
            import os
            env=dict(os.environ, PYTHONPATH=str(self.repo)+os.pathsep+os.environ.get('PYTHONPATH',''))
            try:
                subprocess.run(cmd,cwd=self.repo,env=env,check=True);write(meta,spec)
                self.status[name + ' / triggers'] = 'concluído'
            except subprocess.CalledProcessError as exc:
                self.status[name + ' / triggers'] = f'erro: {exc}'
            self.persist()
        rows=[{'model':n,'trigger_tokens':len(read(self.root/BLOCKS[1]/f'trigger_words_{n}.json'))} for n in BASES if n in self.results and (self.root/BLOCKS[1]/f'trigger_words_{n}.json').exists() and (self.root/BLOCKS[1]/f'trigger_words_{n}.manifest.json').exists() and read(self.root/BLOCKS[1]/f'trigger_words_{n}.manifest.json') == self.results[n]['spec']]
        csv_write(self.root/BLOCKS[1]/'trigger_counts.csv',rows)

    def run(self):
        from params import build_parser
        from util import set_seed
        set_seed(build_parser().parse_args([]))
        for name in list(HF_MODELS)+LOCAL_MODELS:self.evaluate(name)
        self.reports();self.recheck();self.persist()
        return self.status
