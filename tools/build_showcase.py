#!/usr/bin/env python3
"""Rebuild README figures from committed experiment JSON; never calls a server."""
import argparse
import json
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'phase3/results'
OUT = ROOT / 'docs/assets'
KV_SOURCES = {
    'normal': ['token_budget_20260923-190102-1790161262773589000_4096_short_formal.json',
               'token_budget_20260923-190152-1790161312720515000_4096_short_formal.json',
               'token_budget_20260923-190232-1790161352133941000_4096_short_formal.json'],
    '1024_blocks': ['token_budget_20260923-190501-1790161501866669000_4096_short_formal.json',
                    'token_budget_20260923-190612-1790161572382679000_4096_short_formal.json',
                    'token_budget_20260923-190655-1790161615006383000_4096_short_formal.json'],
}


def stats(values):
    if len(values) != 3:
        raise ValueError(f'Expected three formal runs, got {len(values)}')
    return {'median': median(values), 'min': min(values), 'max': max(values), 'runs': len(values)}


def collect():
    pairs = [(p, json.loads(p.read_text())) for p in sorted(RESULTS.glob('*.json'))]
    data = {'experiment_date': '2026-09-23', 'aggregation': 'median and min-max of three formal runs; warmups excluded',
            'open_loop': {}, 'long_prefill': {}, 'kv_pressure': {}}
    for rate in [1, 2, 4, 6]:
        rows = [(p,d) for p,d in pairs if p.name.startswith('load_curve_20260923-')
                and not d.get('warmup') and d.get('mode') == 'open'
                and d.get('offered_rate_req_s') == rate and d.get('fixed_output_tokens') == 64
                and d.get('requests') == 12]
        for _,d in rows:
            if d['slo'] != {'client_ttft_s_lte': 0.5, 'client_e2e_s_lte': 5.0}:
                raise ValueError('SLO drift')
        data['open_loop'][str(rate)] = {
            'completed_rps': stats([d['summary']['achieved_request_throughput_req_s'] for _,d in rows]),
            'goodput_rps': stats([d['summary']['slo_goodput_req_s'] for _,d in rows]),
            'sources': [str(p.relative_to(ROOT)) for p,_ in rows]}
    for budget in [512, 4096]:
        rows = [(p,d) for p,d in pairs if p.name.startswith('token_budget_20260923-')
                and not d.get('warmup') and d.get('shape') == 'long'
                and d.get('declared_max_num_batched_tokens') == budget
                and d.get('fixed_output_tokens') == 128 and d.get('requests') == 6]
        data['long_prefill'][str(budget)] = {
            'server_mean_ttft_s': stats([d['summary']['histogram_deltas']['vllm:time_to_first_token_seconds']['mean_s'] for _,d in rows]),
            'sources': [str(p.relative_to(ROOT)) for p,_ in rows]}
    for group, names in KV_SOURCES.items():
        rows = [json.loads((RESULTS/n).read_text()) for n in names]
        for d in rows:
            if d['warmup'] or d['fixed_output_tokens'] != 2048 or d['requests'] != 8:
                raise ValueError('KV experiment configuration drift')
        # First take the worst per-request content gap in each batch, then the
        # median across batches. This is not a token-level ITL percentile.
        maxima = [max(max(r['content_event_gaps_s']) for r in d['results'] if r['ok']) for d in rows]
        data['kv_pressure'][group] = {
            'batch_max_content_gap_s': stats(maxima),
            'throughput_tok_s': stats([d['summary']['output_throughput_tok_s'] for d in rows]),
            'preemptions': [d['summary']['num_preemptions_delta'] for d in rows],
            'sources': [str((RESULTS/n).relative_to(ROOT)) for n in names]}
    return data


def plot(data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
        'axes.spines.top':False,'axes.spines.right':False,'axes.labelcolor':'#58677b',
        'xtick.color':'#58677b','ytick.color':'#58677b','text.color':'#1e2d43',
        'axes.edgecolor':'#cbd5e1','axes.titleweight':'bold','svg.fonttype':'none'})
    fig, axes=plt.subplots(1,3,figsize=(14,4.5))
    fig.patch.set_facecolor('#ffffff')
    colors=['#3677bd','#d87c47']
    def errors(rows):
        return [[r['median']-r['min'] for r in rows],[r['max']-r['median'] for r in rows]]
    ax=axes[0]; rates=[1,2,4,6]
    for metric,label,color,offset in [('completed_rps','Completed requests',colors[0],-.035),('goodput_rps','SLO goodput',colors[1],.035)]:
        rows=[data['open_loop'][str(x)][metric] for x in rates]
        ax.errorbar([x+offset for x in rates],[r['median'] for r in rows],yerr=errors(rows),fmt='o-',capsize=3,label=label,color=color,linewidth=2,markersize=5)
    ax.set(title='Throughput vs. useful work',xlabel='Offered load (requests/s)',ylabel='Completed / SLO-passing requests/s')
    ax.set_xticks(rates); ax.set_ylim(bottom=0);ax.legend(loc='upper left',frameon=False,fontsize=9)
    ax=axes[1];rows=[data['long_prefill'][str(x)]['server_mean_ttft_s'] for x in [512,4096]]
    ax.bar(['512','4096'],[r['median'] for r in rows],yerr=errors(rows),color=colors,capsize=4,width=.52,zorder=3)
    ax.set(title='Long-input prefill',xlabel='Token budget per scheduler step',ylabel='Server mean TTFT (s)');ax.set_ylim(0,.72)
    for i,r in enumerate(rows):ax.text(i,r['max']+.025,f"{r['median']:.4f}s",ha='center',fontsize=11)
    ax=axes[2];rows=[data['kv_pressure'][key]['batch_max_content_gap_s'] for key in ['normal','1024_blocks']]
    ax.bar(['Normal KV','1024 blocks'],[r['median'] for r in rows],yerr=errors(rows),color=colors,capsize=4,width=.52,zorder=3)
    ax.set(title='Preemption and stream stalls',ylabel='Worst client content gap per batch (s)');ax.set_ylim(0,3.8)
    for i,r in enumerate(rows):ax.text(i,r['max']+.14,f"{r['median']:.3f}s",ha='center',fontsize=11)
    for ax in axes:
        ax.grid(axis='y',alpha=.22,zorder=0);ax.yaxis.set_major_locator(MaxNLocator(5))
    fig.suptitle('vLLM-Lab  /  Evidence from recorded experiments',x=.06,y=.985,ha='left',fontsize=17,fontweight='bold')
    fig.text(.06,.038,'Qwen3-8B / vLLM 0.29.0 / RTX 4090 48GB | 2026-09-23 | median + min-max, n=3 per point',fontsize=9,color='#58677b')
    fig.text(.06,.003,'Panels use different workloads. Goodput: success + client TTFT <= 0.5s + E2E <= 5s. Client timing includes the SSH path.',fontsize=9,color='#58677b')
    fig.tight_layout(rect=[.02,.075,1,.91],w_pad=2.2)
    fig.savefig(OUT/'performance-evidence.png',dpi=180,bbox_inches='tight',facecolor='white')
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--json-only',action='store_true');args=parser.parse_args()
    data=collect();OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'evidence-summary.json').write_text(json.dumps(data,indent=2)+'\n')
    if not args.json_only:plot(data)
    print('Rebuilt recorded evidence; no model inference or network requests.')

if __name__=='__main__':main()
