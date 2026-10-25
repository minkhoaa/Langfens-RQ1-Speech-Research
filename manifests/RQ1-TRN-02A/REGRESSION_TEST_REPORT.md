# Regression test report

- Local CUDA environment full pytest suite: **89 passed, 0 failed** (one expected warning from a test explicitly exercising invalid CTC lengths). Includes regression fix making the CUDA-only metric test skip only when CUDA is unavailable; CUDA failures still fail the test.
- Targeted post-driver/AP tests: **19 passed**.
- Colab T4 real-audio feasibility: **PASS** for bounded forward/backward, two optimizer steps, AMP at scale 4096, padding lengths, finite losses/gradients, head updates, frozen backbone/diagnosis invariants, and checkpoint reload.
- No test partition audio/labels were loaded. No full training regression or validation test was run.
