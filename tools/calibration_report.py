"""Offline CSV and SVG calibration summaries; never launches a solve."""
import argparse
import csv
from html import escape
from pathlib import Path
import math
from query_adapt.data import read_json


def collect(run):
    rows=[]
    for directory in sorted(Path(run).glob('lambda-*')):
        for path in sorted(directory.glob('*.json')):
            saved=read_json(path)
            if 'history' not in saved:
                continue
            final=saved['history'][-1]
            rows.append({'lambda':float(directory.name[7:]),'label':path.stem,'method':saved['method'],
                         'status':saved['status'],'iterations':final['iteration'],
                         'seconds':saved['costs']['total_seconds'],'certificate_B':final['certificate_B'],
                         'parameter_error_bound':final['parameter_error_bound'],
                         **{key:saved['costs'][key] for key in ('Xv','XTv','hessian_builds','linear_solves','line_search_evaluations')}})
    return rows


def svg_plot(directory, axis, pattern="*.json", title=None):
    traces=[]
    for path in sorted(directory.glob(pattern)):
        saved=read_json(path)
        if 'history' not in saved:
            continue
        points=[(r[axis],r['certificate_B']) for r in saved['history'] if r.get('certificate_B') is not None and r['certificate_B']>0 and math.isfinite(r['certificate_B'])]
        if points: traces.append((path.stem,points))
    if not traces: raise ValueError('no finite positive certificate history')
    xmax=max(p[0] for _,points in traces for p in points) or 1
    ys=[math.log10(p[1]) for _,points in traces for p in points]
    lo,hi=min(min(ys),-12),max(max(ys),-9)
    if hi==lo: hi+=1
    colors=['#1479b8','#d45e00','#009d73','#cc79a7','#7752a8','#777777','#d4a000','#222222']
    title=title or f'{directory.name}: calibration certificate, no formal timing'
    parts=['<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="650" viewBox="0 0 1000 650">','<rect width="1000" height="650" fill="white"/>',f'<text x="70" y="28" font-size="19">{escape(title)}</text>']
    def px(x):return 75+650*x/xmax
    def py(y):return 550-480*(math.log10(y)-lo)/(hi-lo)
    for y in range(math.floor(lo),math.ceil(hi)+1):
        yy=py(10.**y)
        parts.append(f'<path d="M75 {yy:.3f} H725" stroke="#dddddd"/><text x="5" y="{yy+4:.3f}" font-size="12">1e{y}</text>')
    for x in range(6):
        value=x*xmax/5
        parts.append(f'<text x="{px(value):.3f}" y="577" font-size="12">{value:.3g}</text>')
    parts.append(f'<text x="320" y="610" font-size="16">{axis} (calibration trajectory)</text>')
    for i,(label,points) in enumerate(traces):
        color=colors[i%len(colors)]
        coords=' '.join(f'{px(x):.3f},{py(y):.3f}' for x,y in points)
        parts.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="1.5"/><text x="755" y="{80+i*24}" fill="{color}" font-size="14">{escape(label)}</text>')
    parts.append('</svg>')
    return '\n'.join(parts)


def report(run, output):
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    rows=collect(run)
    if not rows: raise ValueError('no saved solver histories')
    with (output/'calibration.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    for directory in sorted(Path(run).glob('lambda-*')):
        for axis in ('iteration','seconds'):
            (output/f'{directory.name}-{axis}.svg').write_text(svg_plot(directory,axis))
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    print(f'Saved {len(report(args.run,args.output))} trajectories; calibration only.')

if __name__=='__main__':main()
