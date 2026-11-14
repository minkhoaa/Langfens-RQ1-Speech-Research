# M5 control repair before inspecting M5 held results

Audit during execution found repeated_real used the first300 eligible FIT IDs
cyclically, while pseudo adds50 examples for each of6FIT speakers. Original
control thus changes speaker exposure as well as audio/label source. Keep its
complete runs; do not use them alone for a causal data benefit claim.

No M5 held metrics have been inspected/pooled when registering this repair.
Add balanced_real at the same seed101/202 ×4fold,16epochs, same model, batch
ordering, optimizer updates and initial weights as pseudo. For every extra
pseudo row, substitute a real eligible FIT example from that row's speaker;
cycle deterministically within that speaker. Identical extra-row speaker
counts and count of base samples, no held speaker adaptation. This is a
control repair, not model/threshold tuning. Publish all original controls.
The repair omits only the extra post-training per-attribute-head diagnostic
sweep (already measured in M4); all training, checkpoint selection, realized
predictions, MDD metrics and posteriors remain identical in scope.

Run only after the frozen M6 handoff, within the original GPU wall3600s cap,
reserving the observed failed-M3 startup wall envelope. If insufficient time,
report the confounding and leave M5 attribution unfulfilled; never conceal it.
Do not change the preregistered M6 joint candidate from M5 held results.
