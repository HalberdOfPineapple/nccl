Geometry is CTAs × threads/CTA; arrows show H → P when the configurations differ. Zero co-runner CTAs means inactive.

| Injection | Communication geometry (CTAs × threads/CTA) | Co-runner geometry (CTAs × threads/CTA) | Pairs | Healthy throughput (GB/s) | Perturbed throughput (GB/s) | Paired throughput change | Throughput effect / healthy IQR | Issue interval H → P (µs) | Ordering interval H → P (µs) | Achieved co-runner rate |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| compute | 32 × 256 | 0 × 256 → 8 × 256 | 5 | 82.06 | 82.10 | +0.04% | 1.25 | 3170.82 → 3164.29 | 9.96 → 10.30 | 1.86 |
| compute | 32 × 256 | 0 × 256 → 32 × 256 | 5 | 82.02 | 83.23 | +1.50% | 5.14 | 3173.45 → 3149.18 | 10.61 → 10.14 | 7.43 |
| compute | 32 × 256 | 0 × 256 → 100 × 256 | 5 | 81.89 | 83.22 | +1.60% | 15.75 | 3180.00 → 3157.77 | 10.54 → 10.10 | 23.11 |
| source memory | 32 × 256 | 0 × 256 → 8 × 256 | 5 | 82.06 | 81.78 | -0.41% | -5.03 | 3172.88 → 3160.84 | 9.77 → 9.70 | 20.48 |
| source memory | 32 × 256 | 0 × 256 → 32 × 256 | 5 | 82.04 | 82.32 | +0.50% | 1.86 | 3172.22 → 3148.78 | 10.10 → 9.40 | 82.42 |
| source memory | 32 × 256 | 0 × 256 → 100 × 256 | 5 | 82.05 | 84.44 | +2.97% | 31.17 | 3171.98 → 3114.66 | 10.10 → 8.02 | 259.00 |
| destination memory | 32 × 256 | 0 × 256 → 8 × 256 | 5 | 82.07 | 82.04 | -0.03% | -1.40 | 3172.87 → 3172.73 | 9.92 → 10.24 | 20.31 |
| destination memory | 32 × 256 | 0 × 256 → 32 × 256 | 5 | 82.02 | 82.00 | -0.09% | -0.78 | 3173.08 → 3172.36 | 10.56 → 9.79 | 79.34 |
| destination memory | 32 × 256 | 0 × 256 → 100 × 256 | 5 | 82.00 | 82.02 | -0.00% | 0.13 | 3172.80 → 3173.39 | 10.66 → 9.86 | 246.74 |
| vector source memory | 32 × 256 | 0 × 256 → 8 × 256 | 5 | 80.60 | 80.73 | +0.16% | 3.74 | 3241.06 → 3218.48 | 1.70 → 1.71 | 77.04 |
| vector source memory | 32 × 256 | 0 × 256 → 32 × 256 | 5 | 80.45 | 81.50 | +1.40% | 14.13 | 3248.34 → 3194.74 | 1.70 → 1.71 | 308.42 |
| vector source memory | 32 × 256 | 0 × 256 → 100 × 256 | 5 | 80.46 | 80.29 | -0.14% | -2.54 | 3248.57 → 3255.41 | 1.70 → 1.75 | 934.12 |
| vector destination memory | 32 × 256 | 0 × 256 → 8 × 256 | 5 | 80.61 | 80.59 | -0.01% | -0.39 | 3240.78 → 3240.02 | 1.70 → 1.70 | 74.64 |
| vector destination memory | 32 × 256 | 0 × 256 → 32 × 256 | 5 | 80.58 | 80.62 | +0.06% | 0.66 | 3241.14 → 3240.52 | 1.70 → 1.70 | 289.37 |
| vector destination memory | 32 × 256 | 0 × 256 → 100 × 256 | 5 | 80.55 | 80.57 | +0.08% | 0.14 | 3242.14 → 3241.70 | 1.70 → 1.70 | 850.36 |
| vector destination memory | 32 × 256 | 0 × 256 → 131 × 256 | 5 | 80.46 | 80.59 | +0.07% | 1.52 | 3246.80 → 3241.07 | 1.70 → 1.70 | 1082.33 |
| instrumentation overhead | 32 × 256 | 0 × 256 | 5 | 82.06 | 82.03 | -0.04% | -1.74 | unavailable → 3172.63 | unavailable → 9.76 | 0.00 |
| vector instrumentation overhead | 32 × 256 | 0 × 256 | 5 | 80.57 | 80.60 | +0.05% | 1.19 | unavailable → 3241.16 | unavailable → 1.70 | 0.00 |
