#!/usr/bin/env python3
"""Report paired communication outcomes before destination workload outcomes."""
import json
import math
import statistics
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from collect import summarize, percentile, write_csv
from plot_style import zero_based_upper
from run_saturated_contention import RESULTS, LEVELS


def median(rows, field):
    return statistics.median(row[field] for row in rows)


def load(label):
    return summarize(RESULTS/'raw'/f'{label}.json')


def compare(prefix, field):
    healthy_rows = [load(f'{prefix}_H{repetition}') for repetition in range(1, 6)]
    pressure_rows = [load(f'{prefix}_P{repetition}') for repetition in range(1, 6)]
    healthy = [median(rows, field) for rows in healthy_rows]
    pressure = [median(rows, field) for rows in pressure_rows]
    healthy_iqr = percentile(healthy, .75)-percentile(healthy, .25)
    paired_percent = [(perturbed/reference-1)*100 for reference, perturbed in zip(healthy, pressure)]
    return dict(healthy=statistics.median(healthy), pressure=statistics.median(pressure),
                healthy_min=min(healthy), healthy_max=max(healthy), healthy_iqr=healthy_iqr,
                pressure_min=min(pressure), pressure_max=max(pressure),
                paired_percent=statistics.median(paired_percent), paired_min=min(paired_percent), paired_max=max(paired_percent),
                effect_iqr=(statistics.median(pressure)-statistics.median(healthy))/healthy_iqr if healthy_iqr else None,
                slower_pairs=sum(perturbed<reference for reference, perturbed in zip(healthy, pressure)),
                healthy_p05=percentile([row[field] for rows in healthy_rows for row in rows], .05),
                pressure_p05=percentile([row[field] for rows in pressure_rows for row in rows], .05),
                healthy_worst=min(row[field] for rows in healthy_rows for row in rows),
                pressure_worst=min(row[field] for rows in pressure_rows for row in rows))


def main():
    (RESULTS/'figures').mkdir(exist_ok=True)
    intervals = []
    total_windows = 0
    total_transfers = 0
    for path in sorted((RESULTS/'raw').glob('*.json')):
        data = json.loads(path.read_text())
        if 'ranks' not in data:
            continue
        assert data['verified']
        rows = summarize(path)
        total_windows += len(rows)
        total_transfers += len(rows) if data['communication'] else 0
        for rank in data['ranks']:
            for records in rank['iterations']:
                assert len({record['sm_id'] for record in records}) == len(records)
                for record in records:
                    if record['role'] == 1:
                        assert record['work_units'] > 0 and math.isfinite(record['checksum'])
        if data['co_ctas']:
            assert min(row['minimum_overlap_fraction'] for row in rows) == 1
            assert min(row['co_work_units'] for row in rows) > 0
        telemetry = json.loads(path.with_suffix('.telemetry.json').read_text())
        intervals.append((telemetry['host_start_ns'], telemetry['host_end_ns'], path.stem))
    intervals.sort()
    assert all(previous[1] <= following[0] for previous, following in zip(intervals, intervals[1:])), 'Overlapping invocations'
    communication = []
    reads = []
    phases = []
    for memory in LEVELS:
        communication.append(dict(read_ctas=memory, **compare(f'primary_m{memory}', 'throughput_gbps')))
        solo = [median(load(f'calibration_m{memory}_R{repetition}'), 'co_rate') for repetition in range(1, 6)]
        read = dict(read_ctas=memory, **compare(f'mutual_m{memory}', 'co_rate'))
        read['calibration_gbps'] = statistics.median(solo)
        for arm, name in [('H', 'solo'), ('P', 'concurrent')]:
            rows = [row for repetition in range(1,6) for row in load(f'mutual_m{memory}_{arm}{repetition}')]
            read[f'{name}_active_us'] = median(rows, 'co_active_us')
        reads.append(read)
        for field in ['completed_us', 'issue_us', 'ordering_us', 'notification_us', 'ack_wait_us', 'issue_spread_us', 'receiver_wait_us']:
            phases.append(dict(read_ctas=memory, metric=field, **compare(f'primary_m{memory}', field)))
    overhead = []
    for repetition in range(1,6):
        minimal = median(load(f'probe_d0_R{repetition}'), 'completed_us')
        detailed = median(load(f'probe_d1_R{repetition}'), 'completed_us')
        overhead.append((detailed/minimal-1)*100)
    summary = dict(communication=communication, reads=reads, phases=phases,
                   invocations=len(intervals), retained_windows=total_windows, retained_transfers=total_transfers,
                   probe_overhead_percent=statistics.median(overhead), probe_overhead_range=[min(overhead),max(overhead)],
                   opening_gbps=median(load('opening_reference'),'throughput_gbps'),
                   closing_gbps=median(load('closing_reference'),'throughput_gbps'))
    (RESULTS/'analysis_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    write_csv(RESULTS/'communication.csv', communication)
    write_csv(RESULTS/'reads.csv', reads)
    write_csv(RESULTS/'phases.csv', phases)
    labels = [f'{memory} × 512' for memory in LEVELS]
    figure, axes = plt.subplots(1, 2, figsize=(12,4), layout='constrained')
    for axis, observations, ylabel, title, upper in [
            (axes[0], communication, 'Completed communication (GB/s)', 'Communication throughput', 450),
            (axes[1], reads, 'Logical destination reads (GB/s)', 'Destination read progress', 2500)]:
        for offset, arm, color, label in [(-.18,'healthy','#2878b5','Copy only' if observations is communication else 'Reads only'),
                                         (.18,'pressure','#e1812c','Copy + reads')]:
            values = [row[arm] for row in observations]
            errors = [[row[arm]-row[f'{arm}_min'] for row in observations], [row[f'{arm}_max']-row[arm] for row in observations]]
            axis.bar([position+offset for position in range(4)],values,width=.34,color=color,label=label,yerr=errors,capsize=3)
        axis.set(xticks=range(4),xticklabels=labels,xlabel='Destination reads: CTAs × threads/CTA',ylabel=ylabel,title=title,ylim=(0,upper))
        axis.legend()
        axis.spines[['top','right']].set_visible(False)
    figure.suptitle('Source copy: 128 × 512; 2048 MiB; batch 8; CTA-local stride\nDestination control: 1 × 512; five paired invocations per condition')
    figure.savefig(RESULTS/'figures/throughput.png',dpi=150)
    plt.close(figure)
    figure, axes = plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for axis,field,title in zip(axes,['issue_us','ordering_us','ack_wait_us'],['Source issue loop','Source ordering interval','Source acknowledgement wait']):
        upper = zero_based_upper([row[arm] for row in phases if row['metric'] == field for arm in ['healthy', 'pressure']])
        for arm,color,label in [('healthy','#2878b5','Copy only'),('pressure','#e1812c','Copy + reads')]:
            values=[next(row[arm] for row in phases if row['read_ctas']==memory and row['metric']==field) for memory in LEVELS]
            axis.plot(labels,values,'o-',color=color,label=label,linewidth=1)
        axis.set(title=title,ylabel='Median CTA duration (µs)',xlabel='Destination reads: CTAs × threads/CTA',ylim=(0,upper))
        axis.spines[['top','right']].set_visible(False)
    axes[0].legend()
    figure.suptitle('Source copy: 128 × 512; 2048 MiB; batch 8; CTA-local stride; destination control: 1 × 512\nPhase medians do not add up to completed-transfer duration')
    figure.savefig(RESULTS/'figures/phases.png',dpi=150)
    plt.close(figure)
    write_report(summary)
    print(json.dumps(summary, indent=2))


def write_report(summary):
    communication = summary['communication']
    reads = summary['reads']
    strongest = communication[-1]
    strongest_reads = reads[-1]
    tail = next(row for row in summary['phases'] if row['read_ctas'] == 131 and row['metric'] == 'ack_wait_us')
    issue = next(row for row in summary['phases'] if row['read_ctas'] == 131 and row['metric'] == 'issue_us')
    lines = ['# Destination reads with the calibrated communication kernel', '',
             '**Question.** Does destination read traffic slow the communication kernel after its request supply reaches the measured throughput plateau?', '',
             f"**Answer.** With communication fixed at 128 CTAs × 512 threads/CTA, 2048 MiB and batch 8, median paired communication changes range from {min(row['paired_percent'] for row in communication):+.3f}% to {max(row['paired_percent'] for row in communication):+.3f}%. All 20 primary pairs show a small decrease, but none establishes substantial communication degradation. At destination reads 131 × 512, the separately paired read rate changes by {strongest_reads['paired_percent']:+.2f}%. A practical program plateau does not identify the physical bottleneck.", '',
             '[Plan](../../04_shared_resource_contention.md) · [Communication calibration](../2026-10-27-saturation-gpu6-7/README.md) · [Configurations](../../configs/saturated_contention_gpu6_7/) · [Analysis script](../../../../common/scripts/analyze_saturated_contention.py)', '',
             '| Meaning | Registry |', '|---|---|', '| Completed communication duration / throughput | A1 |', '| Source issue, ordering, publication, acknowledgement and CTA spread | A2–A7 |', '| Destination read work, active duration and SM placement | A8–A10 |', '| Payload correctness | A11 |', '| Timeline and device operating state | B1–B3 |', '',
             '## Communication under destination read pressure', '', '### Motivation', '',
             'Earlier destination-pressure tests used communication settings whose throughput still increased with more issuing threads. This comparison uses the separately confirmed plateau and asks whether destination pressure now reduces communication performance.', '', '### Setup', '',
             '| Knob | Role | Setting |', '|---|---|---|',
             '| Endpoint pair | Constant | Physical GPU 6 → GPU 7; H100 80 GB; 132 SMs each |',
             '| Source communication | Constant | 128 CTAs × 512 threads/CTA; 2048 MiB; CTA-local stride; batch 8 independent scalar 4-byte operations |',
             '| **Destination read count** | **Independent variable** | **0 (copy-only), 8, 32, 100, 131 CTAs × 512 threads/CTA** |',
             '| Destination control | Constant | 1 CTA × 512 threads/CTA, one polling lane, separate SM from reads |',
             '| Destination reads | Constant | Disjoint local 1 GiB buffer; vector loads; 256 KiB bursts; pacing 0 |',
             '| Residency | Constant | One CTA per SM; 116736 dynamic shared bytes per CTA |',
             '| Completion | Constant | All writers system-fence, CTA publication, destination observation and acknowledgement |',
             '| Primary pairs | Constant | Five pairs per read setting; three warm-ups + twelve retained transfers per invocation; rotate level order |',
             '| Read-only calibration | Separate control | Five invocations × five retained 20 ms windows per level; two warm-ups |',
             '| Read-only matched pairs | Separate control | Five pairs per level; three warm-ups + twelve windows; source timer controls 128 × 512, destination control 1 × 512 |',
             '| Probe overhead | Separate control | Five minimal/detailed pairs with reads 131 × 512 |', '', '### Method', '',
             'The binary and source hashes exactly match the successful batch calibration (`build_provenance.json`). No CUDA code, clocks, or power settings changed. Reads-alone calibration precedes primary pairs. Each primary pressure run immediately follows its own copy-only reference. Read performance uses a separate set of adjacent reads-only / reads-with-copy pairs, with the solo timer set to the median concurrent read-active duration measured in the primary runs.', '',
             f"There are {summary['invocations']} accepted timing invocations, {summary['retained_transfers']} retained transfers and {summary['retained_windows']-summary['retained_transfers']} retained read-only windows. Every retained transfer passed full-payload verification. Recorded SM IDs show distinct SMs for all CTAs within each endpoint; every read CTA covers the full destination receiver interval and completes work. Host telemetry intervals confirm no accepted invocations overlap. Per-device locks guard runs. Raw JSON, snapshots and telemetry remain local under ignored `raw/`; reproduction requires those files or rerunning the saved runner.", '',
             'Results use medians of five invocation medians. Percentage changes are medians of the five paired percentage changes, so they need not equal the ratio of displayed grand medians. Reference IQR is across invocation medians. The standardized difference is (pressure median − reference median) / reference IQR; a small denominator can make a tiny practical change look large.', '', '### Results', '',
             '![Communication throughput first; separately paired read progress second](figures/throughput.png)', '',
             'All rows: source communication **128 × 512**, **2048 MiB**, **batch 8**, **CTA-local stride**; destination control **1 × 512**. Geometry notation means CTAs × threads/CTA. Copy-only references have destination reads **0 × 512**.', '',
             '| Destination reads | Copy-only GB/s (min–max) | With reads GB/s | Paired change % (min–max) | Slower pairs | Reference IQR GB/s | Difference / IQR |', '|---|---|---|---|---|---|---|']
    for row in communication:
        lines.append(f"| {row['read_ctas']} × 512 | {row['healthy']:.3f} ({row['healthy_min']:.3f}–{row['healthy_max']:.3f}) | {row['pressure']:.3f} | {row['paired_percent']:+.3f} ({row['paired_min']:+.3f}–{row['paired_max']:+.3f}) | {row['slower_pairs']}/5 | {row['healthy_iqr']:.4f} | {row['effect_iqr']:.2f} |")
    lines += ['', '| Destination reads | Copy-only / with-reads p05 GB/s | Copy-only / with-reads minimum GB/s |', '|---|---|---|']
    for row in communication:
        lines.append(f"| {row['read_ctas']} × 512 | {row['healthy_p05']:.3f} / {row['pressure_p05']:.3f} | {row['healthy_worst']:.3f} / {row['pressure_worst']:.3f} |")
    lines += ['', 'Transfer-level p05 and minima cover 60 retained transfers per arm and condition. They expose slow transfers rather than treating all within-invocation samples as independent repetitions.', '',
              '![Source phase durations](figures/phases.png)', '',
              '| Destination reads | Completed duration µs, only / with reads | Issue-loop µs, only / with reads | Ordering µs, only / with reads | Acknowledgement wait µs, only / with reads |', '|---|---|---|---|---|']
    for memory in LEVELS:
        values = [next(row for row in summary['phases'] if row['read_ctas']==memory and row['metric']==field) for field in ['completed_us','issue_us','ordering_us','ack_wait_us']]
        lines.append(f'| {memory} × 512 | '+' | '.join(f"{row['healthy']:.2f} / {row['pressure']:.2f}" for row in values)+' |')
    lines += ['', 'Issue-loop duration includes source loads, posted stores, and synchronization; it is not individual remote-store latency. Phase medians do not add up to completed duration. Full phase/spread comparisons are in `phases.csv`.', '',
              'Destination read performance follows. These are **separate, duration-matched pairs**, not the copy-only references above. Source communication geometry is still **128 × 512**, payload **2048 MiB**, batch **8**, CTA-local stride; the reads-only arm uses source timer controls **128 × 512** and no copy.', '',
              '| Destination reads | 20 ms calibration GB/s | Matched reads-only GB/s | With copy GB/s | Paired read change % | Actual active duration µs, only / with copy |', '|---|---|---|---|---|---|']
    for row in reads:
        lines.append(f"| {row['read_ctas']} × 512 | {row['calibration_gbps']:.2f} | {row['healthy']:.2f} | {row['pressure']:.2f} | {row['paired_percent']:+.2f} | {row['solo_active_us']:.1f} / {row['concurrent_active_us']:.1f} |")
    lines += ['', 'Actual standalone read intervals are about 19–22 µs longer than concurrent intervals despite matching the requested timer; startup/tail and stop checks remain included. Logical read GB/s counts completed 256 KiB bursts over the union of read-CTA active intervals, including startup/tail. It is not measured physical HBM traffic. Read-only calibration still rises from 100 × 512 to 131 × 512. This tests the maximum available CTA count of the unchanged injector, not an established HBM-saturation point. Filling the available SMs is not proof of HBM saturation.', '',
              f"Detailed probes change completed duration by a median {summary['probe_overhead_percent']:+.3f}% at reads 131 × 512 (five paired invocations; range {summary['probe_overhead_range'][0]:+.3f}% to {summary['probe_overhead_range'][1]:+.3f}%). Opening and closing copy-only references are {summary['opening_gbps']:.3f} and {summary['closing_gbps']:.3f} GB/s.", '',
              '### Interpretation', '',
              f"- **Communication changes little, consistently.** All 20 pairs decrease, with median effects below 0.12%. At reads 131 × 512, throughput moves from {strongest['healthy']:.3f} to {strongest['pressure']:.3f} GB/s. Large IQR-normalized values reflect very stable references, not large practical regressions. These adjacent pairs always put copy-only first, so a systematic order effect is not excluded.",
              f"- **The destination workload shows the larger change.** At reads 131 × 512, logical reads fall from {strongest_reads['healthy']:.2f} to {strongest_reads['pressure']:.2f} GB/s ({strongest_reads['paired_percent']:+.2f}%). Every read CTA completes bursts and covers receiver activity. This is evidence of interaction in the concurrent workload even though communication changes little.",
              f"- **Separate copying from completion waiting.** At reads 131 × 512, median CTA issue time changes from {issue['healthy']:.2f} to {issue['pressure']:.2f} µs, while acknowledgement waiting changes from {tail['healthy']:.2f} to {tail['pressure']:.2f} µs. This suggests checking completion/control progress before blaming a large slowdown in the copy loop. CTA medians cannot precisely apportion total duration or establish a remote-store latency.",
              '- **The physical bottleneck remains unresolved.** No validated simultaneous L2/DRAM hardware-counter collection is available from the earlier profiler attempts. These results cannot identify HBM saturation, arbitration policy, or exact transaction routing. Source phase changes alone do not uniquely locate contention. Candidate explanations include a communication limit elsewhere, unequal sharing of endpoint service, and read-kernel limits that leave capacity for incoming writes. Distinguishing them requires validated traffic/counter evidence or separate controlled changes to the read workload.', '',
              '## Reproduction and available observations', '',
              'Run `python experiments/h100x2/thread_store/common/scripts/run_saturated_contention.py` on allocated, idle physical GPUs 6–7 with the calibrated binary. The runner resumes only matching configs and validates retained outputs; remove neither accepted records nor provenance to force a different binary into this campaign. Run `analyze_saturated_contention.py` using the analysis environment with matplotlib. `runs.csv`, `communication.csv`, `reads.csv`, `phases.csv`, and `analysis_summary.json` are compact derived records. Device telemetry is sampled for whole invocations, including allocation, warm-ups and validation; it cannot establish clock state during individual short transfers.', '']
    environment_path = RESULTS/'environment_summary.json'
    if environment_path.exists():
        environment = json.loads(environment_path.read_text())
        lines += ['| Device | Invocation-wide sampled SM clock range MHz | Maximum sampled power W | Runs with link-counter deltas |', '|---|---|---|---|']
        for device in [6, 7]:
            rows = [row for row in environment if row['gpu'] == device]
            lines.append(f"| GPU {device} | {min(row['sm_clock_min_mhz'] for row in rows):.0f}–{max(row['sm_clock_max_mhz'] for row in rows):.0f} | {max(row['power_max_w'] for row in rows):.2f} | {sum(row['tx_delta_kib'] is not None for row in rows)}/{len(rows)} |")
        lines += ['', 'Raw clock-reason masks in `environment_summary.json` are 0x0 and 0x1. Clock samples span idle/setup and active work; this is not evidence of fixed clocks during communication. Link deltas include warm-ups, setup and verification, so they are not the transferred-payload counter for a retained window.', '']
    timeline = RESULTS/'timeline_summary.json'
    if timeline.exists():
        entries = json.loads(timeline.read_text())
        lines += ['Separate Nsight Systems passes retain endpoint overlap and the intended geometry. Each capture has two warm-ups and one retained transfer with CUDA/NVTX tracing, CPU sampling and context-switch collection disabled, and no replay. Their timing is separate from the primary measurements. The strongest profile retains about the same logical read rate as the unprofiled runs; two single-transfer captures validate overlap, not the small primary effect size.', '', '![Separate endpoint timelines](figures/timelines.png)', '',
                  '| Communication | Destination reads | Profiled communication GB/s | Profiled logical reads GB/s | Read coverage of receiver interval |', '|---|---|---|---|---|']
        for row in entries:
            lines.append(f"| 128 × 512; 2048 MiB; batch 8; CTA-local | {row['read_ctas']} × 512 | {row['throughput_gbps']:.2f} | {row['co_rate']:.2f} | {row['minimum_overlap_fraction']:.2f} |")
        lines += ['', 'Destination control is 1 × 512 in both profiles. Coverage is not applicable to the copy-only profile; its stored zero means no read CTAs. A kernel timeline alone cannot separate the fused read/control roles, so coverage and work counts come from device probes.', '']
    (RESULTS/'README.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
