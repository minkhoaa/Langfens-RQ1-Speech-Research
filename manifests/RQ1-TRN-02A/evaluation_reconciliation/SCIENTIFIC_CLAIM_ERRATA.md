# Scientific claim errata — RQ1-TRN-02A

The original baseline comparison overstates what the current evidence establishes. The reported binary AP is 0.316062 for the selected validation checkpoint, versus 0.303063 for the canonical-phone empirical prior, a difference of +0.012999. This is a descriptive single-fold difference, not evidence of a statistically reliable acoustic contribution. The model's reported best F1 (0.3775) is below the canonical prior's (0.3879), at separately selected operating thresholds.

Until an independent replay, paired matched/mismatched acoustic controls, and uncertainty analysis (with the speaker as the relevant grouping unit) are complete, replace claims such as “significantly outperforms” and “demonstrating authentic acoustic dependence” with: “The saved validation report gives AP 0.316062 for the neural system and 0.303063 for the canonical-phone prior; this descriptive +0.012999 difference does not establish an acoustic contribution.”

This task does not access official test data. The Fold 1 official test partition remains sealed. Do not use gold TextGrid timestamps as model inputs or describe the system as ready for acoustic diagnosis. Canonical-phone provenance must be audited separately: the model receives canonical phone indices through the inference interface, while scoring annotations are distinct targets. The baseline comparison's description says its prior is learned from Fold 1 training labels; that provenance should be verified from its reproducible implementation before accepting the baseline as independent.

No significance test, confidence interval, or paired control has been computed in this recovery.
