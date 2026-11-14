# RC2 Gold-Only baseline control — immutable preflight

## Runtime and job gate

- Git HEAD at start: `aa4e6ea2147180a4a820d77a71911f333fcb6721`.
- The kernel is running NVIDIA open-kernel module `615.78.08`; PCI `10de:1f9d` is bound to `nvidia`; `/proc/driver/nvidia/gpus/0000:01:00.0/information` identifies GeForce GTX 1650.
- Failure cause: the sandbox-visible `/dev` initially had no `/dev/nvidiactl`, `/dev/nvidia0`, `/dev/nvidia-uvm` or `/dev/nvidia-modeset`, despite the loaded/attached kernel driver. `nvidia-modprobe -c 0 -m -u` created these nodes in the authorized elevated execution context. `nvidia-smi` then worked in that context. Ordinary sandbox execution still hides the nodes, so CUDA commands must use the approved elevated Phase 3 lease-wrapper route.
- CUDA smoke test `RC2-CUDA-SMOKE-001` passed through `coordination/phase3_gpu_job.py`: PyTorch `2.6.0+cu124`, CUDA 12.4, GTX 1650, FP16 1024×1024 matmul finite, peak allocated 17,956,864 bytes. Wrapper reported 2 MiB device usage and 0% utilization; job exited 0 and lease was released.
- Phase 4/job check before baseline: no Phase 4 process and no running GPU process in smoke-test telemetry. Existing empty Phase 3 `.lock` files were not considered lease ownership; the wrapper acquires advisory shared/global and slot locks.

## Frozen source and data fingerprints

| Input | SHA-256 |
|---|---|
| `research/student/train_light_student.py` | `80c506b8118d4e210ba75e66674d31fd041481c99de775482f479104983ce11b` |
| `data/derived/l2arctic/features/train_features.npz` | `e85cc2844c5b5b785e9f3b8b9b424ec60c5979829a1511a6fd804d1761ec9fd1` |
| `data/derived/l2arctic/features/dev_cal_features.npz` | `29af1e5781c8c36dc854232739beeb072516d88e45fd8bac9a11ce4508c710be` |
| `data/derived/l2arctic/features/dev_search_features.npz` | `8d6a208e6778ae3448aa4ef5b550cd56f3db2af6ef3511a5f0bd3417128e27e3` |
| `data/derived/l2arctic/manifests/train.jsonl` | `1f4da8518d9b184fb737b93c7887ecd7c3e73d6e83260e54651074831f61bcff` |
| `data/derived/l2arctic/manifests/dev_cal.jsonl` | `87fa9e2af6a2ad14d36e7680b977b1789fd20cfbfb56f239d5f840befce21ad6` |
| `data/derived/l2arctic/manifests/dev_search.jsonl` | `a4ab6c7df7bd65b5d213d306c2fdfde6c316cbf7768a3eb63d056775351f99d2` |

## Output isolation

The Phase 3 trainer hardcodes `models/student/{arm}_seed{seed}.pt` and prediction names under `experiments/student_training/`. This baseline therefore uses unique arm name `RC2-GOLD-ONLY-CONTROL`, seed 101, 12 epochs, preserving all published `STU-01-*` checkpoints and predictions. Expected outputs are:

- `models/student/RC2-GOLD-ONLY-CONTROL_seed101.pt`
- `experiments/student_training/RC2-GOLD-ONLY-CONTROL_seed101_dev_cal.npz`
- `experiments/student_training/RC2-GOLD-ONLY-CONTROL_seed101_dev_search.npz`

The run command must go through the Phase 3 lease wrapper; no CPU fallback is allowed. Record wrapper telemetry, stdout, checkpoint and prediction hashes separately after successful completion.
