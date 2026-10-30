#!/usr/bin/env python3
"""Summaries use invocation medians as independent units, retaining H/P pairing."""
import argparse
import csv
import json
from pathlib import Path
import statistics
from collect import percentile


def numeric_median(rows,key):
    return statistics.median(float(row[key]) for row in rows)


def launch_geometry(rows, count_field):
    geometries = sorted({(int(float(row[count_field])), int(float(row['threads']))) for row in rows})
    return '; '.join(f'{count} × {threads}' for count, threads in geometries)


def paired_geometry(healthy_rows, perturbed_rows, count_field):
    healthy = launch_geometry(healthy_rows, count_field)
    perturbed = launch_geometry(perturbed_rows, count_field)
    return healthy if healthy == perturbed else f'{healthy} → {perturbed}'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('directory',type=Path)
    arguments=parser.parse_args()
    rows=list(csv.DictReader((arguments.directory/'runs.csv').open()))
    groups=[]
    geometry_labels={}
    for name in ['compute','source_memory','destination_memory','vector_source_memory','vector_destination_memory','instrumentation_overhead','vector_instrumentation_overhead']:
        levels=[8,32,100,131] if 'instrumentation_overhead' not in name else [None]
        for level in levels:
            prefix=f'{name}_{level}' if level is not None else name
            healthy={int(row['run'].rsplit('_',1)[1]):row for row in rows if row['run'].startswith(prefix+'_H_')}
            perturbed={int(row['run'].rsplit('_',1)[1]):row for row in rows if row['run'].startswith(prefix+'_P_')}
            common_pairs=sorted(healthy.keys() & perturbed.keys())
            if not common_pairs:
                continue
            healthy_rows=[healthy[index] for index in common_pairs]
            perturbed_rows=[perturbed[index] for index in common_pairs]
            geometry_labels[(name,level)] = (
                paired_geometry(healthy_rows, perturbed_rows, 'ctas'),
                paired_geometry(healthy_rows, perturbed_rows, 'co_ctas'),
            )
            group={'name':name,'co_ctas':level,'pairs':len(common_pairs),'metrics':{}}
            for field in ['throughput_gbps','completed_us','issue_us','ordering_us','notification_us','ack_wait_us','issue_spread_us','receiver_wait_us','co_rate','co_active_us','minimum_overlap_fraction']:
                healthy_values=[float(row[field]) for row in healthy_rows if row[field]!='']
                perturbed_values=[float(row[field]) for row in perturbed_rows if row[field]!='']
                if len(healthy_values)!=len(common_pairs) or len(perturbed_values)!=len(common_pairs):
                    group['metrics'][field]={'healthy':statistics.median(healthy_values) if healthy_values else None,'perturbed':statistics.median(perturbed_values) if perturbed_values else None,'healthy_iqr':None,'effect_iqr':None,'paired_percent_median':None,'healthy_range':None,'perturbed_range':None}
                    continue
                healthy_median=statistics.median(healthy_values)
                perturbed_median=statistics.median(perturbed_values)
                healthy_iqr=percentile(healthy_values,.75)-percentile(healthy_values,.25)
                paired_percent=[100*(float(perturbed[index][field])/float(healthy[index][field])-1) for index in common_pairs if float(healthy[index][field])]
                group['metrics'][field]={'healthy':healthy_median,'perturbed':perturbed_median,'healthy_iqr':healthy_iqr,'effect_iqr':(perturbed_median-healthy_median)/healthy_iqr if healthy_iqr else None,'paired_percent_median':statistics.median(paired_percent) if paired_percent else None,'healthy_range':[min(healthy_values),max(healthy_values)],'perturbed_range':[min(perturbed_values),max(perturbed_values)]}
            groups.append(group)
    (arguments.directory/'comparisons.json').write_text(json.dumps(groups,indent=2))
    with (arguments.directory/'comparisons.md').open('w') as output:
        output.write('Geometry is CTAs × threads/CTA; arrows show H → P when the configurations differ. Zero co-runner CTAs means inactive.\n\n')
        output.write('| Injection | Communication geometry (CTAs × threads/CTA) | Co-runner geometry (CTAs × threads/CTA) | Pairs | Healthy throughput (GB/s) | Perturbed throughput (GB/s) | Paired throughput change | Throughput effect / healthy IQR | Issue interval H → P (µs) | Ordering interval H → P (µs) | Achieved co-runner rate |\n|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n')
        def display(value):
            return f'{value:.2f}' if value is not None else 'unavailable'
        for group in groups:
            metrics=group['metrics']
            throughput=metrics['throughput_gbps']
            effect=throughput['effect_iqr']
            effect_text=f'{effect:.2f}' if effect is not None else 'undefined'
            communication_geometry, co_runner_geometry = geometry_labels[(group['name'],group['co_ctas'])]
            output.write(f"| {group['name'].replace('_',' ')} | {communication_geometry} | {co_runner_geometry} | {group['pairs']} | {throughput['healthy']:.2f} | {throughput['perturbed']:.2f} | {throughput['paired_percent_median']:+.2f}% | {effect_text} | {display(metrics['issue_us']['healthy'])} → {display(metrics['issue_us']['perturbed'])} | {display(metrics['ordering_us']['healthy'])} → {display(metrics['ordering_us']['perturbed'])} | {metrics['co_rate']['perturbed']:.2f} |\n")
    print(arguments.directory/'comparisons.md')

if __name__=='__main__':
    main()
