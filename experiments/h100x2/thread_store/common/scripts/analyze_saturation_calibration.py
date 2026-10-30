#!/usr/bin/env python3
"""Report accepted calibration runs; rejected overlap attempts are never globbed."""
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collect import summarize,write_csv
from plot_style import set_zero_based_y
from run_saturation_calibration import RESULTS


def median(values):
    return statistics.median(values)


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,row))+' |' for row in rows])


def main():
    runs=[]
    intervals=[]
    for phase,directory in [('concurrency',RESULTS),('payload',RESULTS/'payload'),('batch',RESULTS/'batch')]:
        for path in sorted((directory/'raw').glob('*.json')):
            data=json.loads(path.read_text())
            if not isinstance(data,dict) or 'ranks' not in data:
                continue
            observations=summarize(path)
            assert data['verified'] and len(observations)==12
            telemetry=json.loads(path.with_suffix('.telemetry.json').read_text())
            intervals.append((telemetry['host_start_ns'],telemetry['host_end_ns'],str(path)))
            runs.append(dict(phase=phase,run=path.stem,ctas=data['ctas'],threads=data['threads'],payload_mib=data['bytes']//1048576,batch=data.get('copy_batch',1),traversal=data['global_stride'],detailed=data['detailed'],read_ctas=data['co_ctas'],gbps=median([row['throughput_gbps'] for row in observations]),completed_us=median([row['completed_us'] for row in observations]),issue_us=median([row['issue_us'] for row in observations]) if data['detailed'] else None,ordering_us=median([row['ordering_us'] for row in observations]) if data['detailed'] else None,ack_us=median([row['ack_wait_us'] for row in observations]) if data['detailed'] else None))
    intervals.sort()
    for previous,current in zip(intervals,intervals[1:]):
        if previous[1]>current[0]:
            raise RuntimeError(f'Overlapping accepted invocations: {previous[2]} and {current[2]}')
    write_csv(RESULTS/'runs.csv',runs)
    lookup={(row['phase'],row['run']):row for row in runs}
    def confirmed(count,threads,payload,batch):
        return [lookup[('batch',f'confirm_c{count}_t{threads}_p{payload}_b{batch}_R{repetition}')] for repetition in range(1,6)]
    settings=[(128,512,2048,8),(128,768,2048,8),(128,1024,2048,8),(128,1024,2048,1),(128,1024,2048,2),(128,1024,2048,4),(64,1024,2048,8),(128,1024,4096,8)]
    summaries=[]
    for count,threads,payload,batch in settings:
        samples=confirmed(count,threads,payload,batch)
        summaries.append(dict(ctas=count,threads=threads,payload_mib=payload,batch=batch,gbps=median([row['gbps'] for row in samples]),minimum_gbps=min(row['gbps'] for row in samples),maximum_gbps=max(row['gbps'] for row in samples),completed_us=median([row['completed_us'] for row in samples]),issue_us=median([row['issue_us'] for row in samples]),ordering_us=median([row['ordering_us'] for row in samples]),ack_us=median([row['ack_us'] for row in samples])))
    def change(reference,pressure):
        return median([(second['gbps']/first['gbps']-1)*100 for first,second in zip(confirmed(*reference),confirmed(*pressure))])
    changes=dict(threads_512_to_768=change((128,512,2048,8),(128,768,2048,8)),threads_768_to_1024=change((128,768,2048,8),(128,1024,2048,8)),batch_2_to_4=change((128,1024,2048,2),(128,1024,2048,4)),batch_4_to_8=change((128,1024,2048,4),(128,1024,2048,8)),payload_2048_to_4096=change((128,1024,2048,8),(128,1024,4096,8)))
    thread_rates=[median([row['gbps'] for row in confirmed(128,threads,2048,8)]) for threads in (512,768,1024)]
    spread=(max(thread_rates)/min(thread_rates)-1)*100
    passed=all(value<1 for value in changes.values()) and spread<1
    minimal=[lookup[('batch',f'probe_c128_t512_p2048_b8_d0_R{repetition}')] for repetition in range(1,6)]
    detailed=[lookup[('batch',f'probe_c128_t512_p2048_b8_d1_R{repetition}')] for repetition in range(1,6)]
    probe_change=median([(second['gbps']/first['gbps']-1)*100 for first,second in zip(minimal,detailed)])
    summary=dict(accepted_invocations=len(runs),retained_transfers=12*len(runs),confirmed=summaries,paired_changes_percent=changes,thread_rate_spread_percent=spread,plateau_criterion_passed=passed,probe_paired_change_percent=probe_change,selected=dict(ctas=128,threads=512,payload_mib=2048,batch=8,gbps=thread_rates[0]))
    (RESULTS/'calibration_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    write_csv(RESULTS/'confirmed.csv',summaries)
    figures=RESULTS/'figures'
    figures.mkdir(exist_ok=True)
    figure,axes=plt.subplots(1,3,figsize=(15,4),layout='constrained')
    groups=[([row for row in summaries if row['ctas']==128 and row['payload_mib']==2048 and row['batch']==8],'threads','Threads per communication CTA','128 communication CTAs; batch 8; 2048 MiB'),([row for row in summaries if row['ctas']==128 and row['threads']==1024 and row['payload_mib']==2048],'batch','Scalar elements per thread batch','128 CTAs × 1024 threads/CTA; 2048 MiB'),([row for row in summaries if row['ctas']==128 and row['threads']==1024 and row['batch']==8],'payload_mib','Transfer size (MiB)','128 CTAs × 1024 threads/CTA; batch 8')]
    for axis,(selected,key,label,title) in zip(axes,groups):
        selected=sorted(selected,key=lambda row:row[key])
        axis.errorbar([row[key] for row in selected],[row['gbps'] for row in selected],yerr=[[row['gbps']-row['minimum_gbps'] for row in selected],[row['maximum_gbps']-row['gbps'] for row in selected]],fmt='o-',color='#2878b5',capsize=3)
        axis.set(title=title,xlabel=label,ylabel='Completed communication throughput (GB/s)',xticks=[row[key] for row in selected])
        set_zero_based_y(axis,450)
        axis.spines[['top','right']].set_visible(False)
    figure.suptitle('Communication-alone confirmation on GPUs 6 → 7\nCTA-local stride; 4-byte stores; no destination reads; five invocations per point')
    figure.savefig(figures/'plateau.png',dpi=150)
    plt.close(figure)
    figure,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for axis,phase,key,title in [(axes[0],'concurrency','threads','Initial expanded limits: 128 CTAs, 256 MiB'),(axes[1],'payload','payload_mib','Longer copies: 128 CTAs × 1024 threads/CTA')]:
        selected=[row for row in runs if row['phase']==phase and row['ctas']==128 and row['batch']==1 and row['traversal']==0 and ((phase=='concurrency' and row['run'].startswith('confirm_')) or (phase=='payload' and row['threads']==1024))]
        values=sorted(set(row[key] for row in selected))
        axis.plot(values,[median([row['gbps'] for row in selected if row[key]==value]) for value in values],'o-',color='#2878b5')
        axis.set(title=title,xlabel='Threads per CTA' if key=='threads' else 'Transfer size (MiB)',ylabel='Completed throughput (GB/s)',xticks=values)
        set_zero_based_y(axis,450)
        axis.spines[['top','right']].set_visible(False)
    figure.suptitle('Earlier calibration steps: batch 1; separate builds and fresh references\nLeft: five invocations per point; right: single-invocation screen')
    figure.savefig(figures/'calibration_steps.png',dpi=150)
    plt.close(figure)
    selected=summary['selected']
    report=f'''# Communication saturation calibration — correction

**Result.** The repeated confirmation {'passes' if passed else 'does not pass'} the stated near-plateau criterion. The selected reference is **128 communication CTAs × 512 threads/CTA, 2048 MiB, batch 8, CTA-local stride**, at **{selected['gbps']:.2f} GB/s**. Destination reads are disabled. This is a measured throughput plateau for this copy implementation and completion protocol; it does not identify the physical limiting resource.

The earlier 64-CTA × 512-thread sweep was incomplete calibration. It should not have been followed by the contention grid while throughput was still increasing. The [plan](../../04_shared_resource_contention.md) now makes this calibration a prerequisite. Earlier measurements remain valid for their recorded settings, but are not a saturated reference.

| Observation | Meaning |
|---|---|
| Completed throughput and duration (A1) | Acknowledged source-local interval; primary outcome |
| Issue, ordering, acknowledgement (A3, A4, A6) | Per-CTA phase medians; not individual remote-store latency and not additive |
| Placement and correctness (A10, A11) | Unique CTA/SM records and full-payload verification |

Definitions are in the [common setup](../../../00_common_setup.md).

## Confirming where throughput stops improving

### Motivation

Increase communication demand beyond the earlier software limits and check whether more threads, more independent scalar operations per thread, or longer transfers still improve throughput.

### Setup

| Knob | Role | Configuration |
|---|---|---|
| GPUs | Constant | Physical GPU 6 → GPU 7; H100, NV18 topology; no clock or power-setting changes |
| **Threads per CTA** | **Independent variable** | **512, 768, 1024**, at 128 CTAs, 2048 MiB, batch 8 |
| **Batch size** | **Independent variable** | **1, 2, 4, 8** scalar elements per thread, at 128 CTAs × 1024 threads/CTA and 2048 MiB |
| **Transfer size** | **Independent variable** | **2048 versus 4096 MiB**, at 128 CTAs × 1024 threads/CTA and batch 8 |
| CTA-count check | Separate comparison | 64 versus 128 CTAs × 1024 threads/CTA, 2048 MiB, batch 8 |
| Accesses | Constant | Scalar 4-byte source loads and peer stores; CTA-local stride |
| Batch implementation | Controlled request supply | Load B independent source elements into registers, then issue their B matching scalar stores; no added fences |
| Destination | Constant within each thread group | 0 read CTAs × group threads; 1 control CTA × group threads, with one polling lane |
| Placement | Constant policy | One CTA per SM, checked through occupancy and recorded SM IDs |
| Sampling | Constant | Five rotated rounds per setting; three warm-ups and twelve retained transfers per invocation |
| Compiler | Constant for confirmation | 52 registers, 4100 bytes static shared memory, no spills; one binary for all batch comparisons |

### Method

Completed duration is source earliest start to latest acknowledgement observation, and throughput is useful bytes divided by that duration. All writers fence before publishing completion; destination acknowledgement and full-buffer verification are retained. Detailed runs also record issue, ordering, and acknowledgement intervals and up to eight software issue checkpoints. Those checkpoints are not remote arrival measurements, and non-divisible loop lengths can yield fewer than eight checkpoints.

The practical criterion is less than 1% median gain on two successive increases in issuing threads, confirmed over five rounds; the three thread-count medians must also span less than 1%. Batch-size and transfer-size checks test whether another avoidable request-supply or short-transfer limitation remains. One percent is a calibration tolerance, not a statistical significance test. Compare within the same compiled program and report the size of each change.

### Results

![Confirmed throughput plateau](figures/plateau.png)

The figure uses zero-based shared scales. Points are medians of five invocation medians; error bars show their minimum and maximum.

'''
    report+=table(['Communication CTAs × threads/CTA','Payload MiB','Batch elements/thread','Median GB/s','Invocation range GB/s','Median completed µs'],[(f"{row['ctas']} × {row['threads']}",row['payload_mib'],row['batch'],f"{row['gbps']:.2f}",f"{row['minimum_gbps']:.2f}–{row['maximum_gbps']:.2f}",f"{row['completed_us']:.2f}") for row in summaries])
    labels={'threads_512_to_768':'128 × 512 → 128 × 768; 2048 MiB; batch 8','threads_768_to_1024':'128 × 768 → 128 × 1024; 2048 MiB; batch 8','batch_2_to_4':'128 × 1024; 2048 MiB; batch 2 → 4','batch_4_to_8':'128 × 1024; 2048 MiB; batch 4 → 8','payload_2048_to_4096':'128 × 1024; batch 8; 2048 → 4096 MiB'}
    report+='\n\n'+table(['Controlled change: CTAs × threads/CTA; other settings fixed','Median within-round throughput change'],[(labels[name],f'{value:+.3f}%') for name,value in changes.items()])
    report+=f'''

### Interpretation

- **More issuing threads no longer give a useful gain at the selected settings.** The 512-, 768-, and 1024-thread medians span {spread:.3f}%. The largest positive step in the thread comparison is {max(changes['threads_512_to_768'],changes['threads_768_to_1024']):.3f}%.
- **Additional request batching and a longer transfer also give less than 1% improvement** in the final upper-level comparisons. These are checks against stopping merely because a configured limit was reached.
- **Use the full selected configuration in later tests.** A reference with only the CTA/thread counts but a different payload or batch size is not this calibrated workload. Changing the binary requires fresh references.
- **Physical cause remains unresolved.** These measurements establish a practical throughput plateau. They do not prove HBM saturation or identify a particular NVLink/L2 queue. No hardware-counter result is claimed.

## What had to change before that confirmation

### Motivation

Document why the previous sweep was insufficient and preserve the exploratory steps rather than presenting only the successful endpoint.

### Setup

| Stage | Independent variables | Fixed settings |
|---|---|---|
| Expanded concurrency | CTA counts 64, 96, 128, 132; threads/CTA 512, 768, 896, 1024 in the recorded combinations | 256 MiB, batch 1, no destination reads; traversal explicitly recorded |
| Transfer duration | Payload 256, 512, 1024, 2048 MiB | 128 CTAs × 1024 threads/CTA, batch 1, CTA-local stride |
| Larger-payload cross-checks | 128 CTAs × 512/768 threads; 64 CTAs × 1024 threads; 132 CTAs × 1024 threads with global stride | 2048 MiB, batch 1, no destination reads |
| Final request-supply check | Batch 1/2/4/8 and the complete configurations above | Separate batch-capable binary; fresh controls |

### Method

The communication/control arrays and per-lane checksum storage were expanded safely for 132 CTAs and 1024 threads/CTA. Device block-size and one-CTA-per-SM checks remain enabled. Adding `payload_mib` changed host configuration only: the saved GPU instruction dumps for the concurrency and payload builds are identical. Adding batched scalar operations changed GPU code and registers (54 → 52), so the final comparisons use new same-binary references.

The initial 96- and 132-CTA local-stride settings performed poorly. The existing global-stride option recovered much of that throughput. Local CTA region starts are not equally aligned for those CTA counts; traversal also changes address order, so the screen alone does not isolate an alignment mechanism. The selected 128-CTA setting keeps contiguous regions aligned and avoids treating that work-assignment effect as saturation.

### Results

![Earlier calibration steps](figures/calibration_steps.png)

`runs.csv` retains every accepted invocation's full configuration and issue, ordering, acknowledgement, and completed timings, grouped by binary phase. The transfer-size screen is exploratory (one invocation per point); the final plateau is supported by the separate five-round confirmation above.

### Interpretation

- **Extending the original limits was necessary but insufficient.** The 256 MiB case still had small gains near the enlarged thread limit; larger transfers raised sustained throughput further.
- **The final configuration tests stronger request supply.** Its same-binary comparisons show that further thread and batch increases do not materially raise throughput. Do not attribute every change between the earlier and final builds to batching alone, because compilation also changed.
- **Keep the earlier negative results in their original scope.** They characterize the lower-demand settings. Future contention tests must use fresh controls at the selected plateau and must first calibrate the destination workload's achieved pressure.

## Validation and retained evidence

The final minimal/detailed probe comparison at **128 CTAs × 512 threads/CTA, 2048 MiB, batch 8, no destination reads** uses five interleaved pairs. Median paired throughput change from minimal to detailed probes is **{probe_change:+.3f}%**. Both modes retain boundary and correctness observations; this is not an entirely uninstrumented reference.

Accepted data contains **{len(runs)} invocations and {len(runs)*12} retained transfers**, including the small old/new-binary correctness checks. Every copy passed full-buffer verification and the placement/timing checks. The generated code retains scalar bulk stores; no register spills were reported.

**Rejected scheduling attempt.** I started probe checks while the last confirmation round was still running. Two own jobs overlapped on GPUs 6–7. All eight round-5 confirmation records and every probe record from that attempt were conservatively excluded, preserved under `batch/raw/rejected_overlap/`, and reacquired sequentially. `batch/rejected_runs_summary.json` records the reason and filenames. The wrapper now holds exclusive per-device file locks before its idle check; a host-only test verified that a second invocation is rejected before GPU work starts. A final audit also verified that all {len(runs)} accepted invocation telemetry intervals are non-overlapping. Rejected samples are not included in the figures, tables, or counts above.

Raw JSON, telemetry, logs, source snapshots, instruction dumps, and the rejected attempt remain local and ignored by Git. Configurations, analysis scripts, build hashes, derived summaries, and figures remain versioned. Each phase records its own source/binary hashes. Earlier binaries are retained locally under `common/bin/push_harness_before_*`; reproducing an earlier phase requires that phase's source and binary, not the latest default build.
'''
    (RESULTS/'README.md').write_text(report)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
