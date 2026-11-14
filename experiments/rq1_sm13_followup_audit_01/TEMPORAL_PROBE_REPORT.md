# Spectrotemporal pilot result

**Run:** 2026-11-13. **Population:** 12 TRAIN speakers, 1,800 utterances, original four outer speaker folds. See the [fixed plan](TEMPORAL_PROBE_PREREGISTRATION.md), [code](run_temporal_probe.py), and [machine results](TEMPORAL_PROBE_RESULTS.json). No DEV or TEST labels were used.

## Equal-effort comparison

The 51D control and 51D+222D temporal candidate used the same per-manner HGB model, positive weights, folds, seed, training labels, and fixed O/I streams. The temporal features describe seven short windows around the **predicted** CharSiu center, never a manual boundary. Archived SM13 is a reference score, not an equal-model ablation.

| Manner | Archived SM13 AP | 51D control AP | 51D+temporal AP | 51D control best retrospective recall at precision ≥50%, FWR ≤0.20 | Temporal equivalent |
|---|---:|---:|---:|---:|---:|
| Vowel | 0.2348 | 0.2808 | 0.2946 | 168/2,280 = **7.37%**; 168 TP, 144 FP | 145/2,280 = **6.36%**; 145 TP, 142 FP |
| Stop | 0.1968 | 0.2364 | 0.2367 | 118/862 = **13.69%**; 118 TP, 116 FP | 82/862 = **9.51%**; 82 TP, 81 FP |

The retrospective operating points assign the full remaining warning allowance to **one manner at a time**, so the vowel and stop figures cannot be combined into one policy. Their thresholds are also selected and measured on the same OOF scores; they are optimistic screening results. Temporal AP improved for vowel and was almost unchanged for stop, but low-risk recall was worse than the 51D control in both manners. Under the preregistered rule, **stop this temporal feature recipe**. AP alone did not predict the requested operating behavior.

The 51D control showed a useful exploratory ranking gain over archived SM13. Because it was noticed after looking at these OOF results, its separate [nested speaker transfer check](STATIC_SPECIALIST_NESTED_PREREGISTRATION.md) is development-exposed and cannot be called an independent confirmation.
