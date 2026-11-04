const results = await runs.all([
  {
    key: "agent_a",
    agent: "worker",
    task: "You are Agent A — Precision & Decision-Policy Benchmark for LANGFENS RQ1 Precision Recovery Experiment 01.\n\nWork strictly inside workstreams/agent_a_policy/. Use .venv/bin/python.\n\nYour computational script run_policy_benchmark.py has already executed and generated POLICY_COMPARISON.csv and policy_summary.json.\n\nYour task is to:\n1. Verify the exact numbers in POLICY_COMPARISON.csv and PRECISION_RECALL_FRONTIER.json.\n2. Write the comprehensive, publication-grade DECISION_POLICY_BENCHMARK.md inside workstreams/agent_a_policy/ and copy it to the repository root.\n3. Verify all screening targets against your empirical results.\n4. Deliver POLICY_COMPARISON.csv, PRECISION_RECALL_FRONTIER.json, and DECISION_POLICY_BENCHMARK.md with exact counts and citations."
  },
  {
    key: "agent_b",
    agent: "worker",
    task: "You are Agent B — Stop-Consonant Acoustic Verification for LANGFENS RQ1 Precision Recovery Experiment 01.\n\nWork strictly inside workstreams/agent_b_stops/. Use .venv/bin/python.\n\nYour computational script run_stop_acoustic_verification.py has executed and generated STOP_VERIFIER_RESULTS.json and STOP_ERROR_CASES.jsonl.\n\nYour task is to:\n1. Inspect the results across all 8 stop verifier candidates and error cases.\n2. Write the publication-grade STOP_ACOUSTIC_VERIFICATION.md inside workstreams/agent_b_stops/ and copy it to the repository root.\n3. Detail the physical acoustic evidence (burst, duration, energy), the failure modes, and why stop acoustic verification filters false alarms but also true omissions.\n4. Deliver STOP_ACOUSTIC_VERIFICATION.md, STOP_VERIFIER_RESULTS.json, and STOP_ERROR_CASES.jsonl with exact numbers and hashes."
  },
  {
    key: "agent_c",
    agent: "worker",
    task: "You are Agent C — Rhotic & Annotation Reliability Investigation for LANGFENS RQ1 Precision Recovery Experiment 01.\n\nWork strictly inside workstreams/agent_c_rhotic/. Use .venv/bin/python.\n\nYour computational script run_rhotic_and_reliability_analysis.py has executed and generated rhotic_summary.json and adjudicated_metrics_summary.json.\n\nYour task is to:\n1. Write RHOTIC_ERROR_ANALYSIS.md inside workstreams/agent_c_rhotic/ and copy to root.\n2. Write ANNOTATION_RELIABILITY_RECONCILIATION.md inside workstreams/agent_c_rhotic/ and copy to root.\n3. Write ADJUDICATED_METRICS.md inside workstreams/agent_c_rhotic/ and copy to root.\n4. Provide the complete phonetic explanation for /r/ false alarms (NJS vs RRBI, tap [ɾ] vs approximant [ɹ], duration, formants, Wav2Vec2 prior) and the Horvitz-Thompson weighted estimates on the 230-case review ledger."
  },
  {
    key: "agent_d",
    agent: "worker",
    task: "You are Agent D — Controlled Acoustic Verifier & System-Level Benchmark for LANGFENS RQ1 Precision Recovery Experiment 01.\n\nWork strictly inside workstreams/agent_d_verifier/. Use .venv/bin/python.\n\nYour computational script run_system_level_benchmark.py has executed and generated SYSTEM_LEVEL_COMPARISON.json.\n\nYour task is to:\n1. Inspect SYSTEM_LEVEL_COMPARISON.json covering the 5 primary configurations and the 4 ablation variants across all 9,956 gold canonical positions.\n2. Write ACOUSTIC_VERIFIER_BENCHMARK.md inside workstreams/agent_d_verifier/ and copy to root.\n3. Write VERIFIER_ABLATION_REPORT.md inside workstreams/agent_d_verifier/ and copy to root.\n4. Detail the full system performance, multi-class MDD invariants, computational overhead (ms/utterance), and why acoustic verification alone fails to achieve the 30% precision screening target on original gold."
  }
]);

return results.map(r => ({ key: r.key, ok: r.ok, error: r.error }));
