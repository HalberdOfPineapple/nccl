#!/usr/bin/env python3
"""Derive whole-invocation environmental context, without claiming phase alignment."""
import argparse
import csv
import io
import json
from pathlib import Path
import re
import statistics


def link_counts(text):
    counts={}
    current_device=None
    for line in text.splitlines():
        device=re.match(r'GPU (\d+):',line)
        if device:
            current_device=int(device.group(1))
        counters=re.search(r'Data (Tx|Rx): (\d+) KiB',line)
        if current_device is not None and counters:
            accumulated=counts.setdefault(current_device,[0,0])
            accumulated[0 if counters.group(1)=='Tx' else 1]+=int(counters.group(2))
    return counts


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('directory',type=Path)
    arguments=parser.parse_args()
    results=[]
    for path in sorted(arguments.directory.glob('*.telemetry.json')):
        run=path.name.removesuffix('.telemetry.json')
        telemetry=json.loads(path.read_text())
        before=json.loads((arguments.directory/f'{run}.before.json').read_text())
        after=json.loads((arguments.directory/f'{run}.after.json').read_text())
        before_links=link_counts(before['link']['stdout'])
        after_links=link_counts(after['link']['stdout'])
        samples={}
        for sample in telemetry['samples']:
            if sample['returncode']:
                continue
            for row in csv.reader(io.StringIO(sample['stdout'])):
                samples.setdefault(int(row[0]), []).append(row)
        for device,observations in samples.items():
            if not observations:
                continue
            def observed(index):
                values=[]
                for observation in observations:
                    try:
                        values.append(float(observation[index]))
                    except ValueError:
                        pass
                return values
            clocks=observed(2)
            power=observed(4)
            entry={'run':run,'gpu':device,'samples':len(observations),'sm_clock_min_mhz':min(clocks) if clocks else None,'sm_clock_max_mhz':max(clocks) if clocks else None,'power_max_w':max(power) if power else None,'clock_reason_masks':sorted(set(row[8].strip() for row in observations)),'tx_delta_kib':None,'rx_delta_kib':None}
            if device in before_links and device in after_links:
                entry['tx_delta_kib']=after_links[device][0]-before_links[device][0]
                entry['rx_delta_kib']=after_links[device][1]-before_links[device][1]
            results.append(entry)
    (arguments.directory/'environment_summary.json').write_text(json.dumps(results,indent=2))
    print(f'{len(results)} device/run environmental records')

if __name__=='__main__':
    main()
