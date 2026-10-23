# LANGFENS — RQ1-STD-01: Standardized Data Contract & Dataloader Schema Specification

**Execution Gate:** RQ1-STD-01 — Data Standardization & Evaluation Protocol Specification  
**Focus Area:** Task F — Schema Design for Standardized Examples, Dataloader Contracts & Provenance Records  
**Author / Engineering Lead:** Senior Speech ML Research Engineer & Research Data Architect  
**Schema Version:** `1.0.0` (Semantic Versioning)  
**Execution State:** SPECIFICATION ONLY (Data Contracts Designed; Preprocessing Execution Deferred)  
**Deliverable Associated Path:** `manifests/RQ1-STD-01/DATA_CONTRACT_SPEC.md`  
**Synced Location:** `kltn_gdrive:KLTN-Dataset/manifests/RQ1-STD-01/`  
**Source Reports Referenced:**
- `manifests/RQ1-AUD-01/reconciliation/MDD_LABEL_ACCOUNTING_SPEC.md`
- `manifests/RQ1-AUD-01/reconciliation/AUDIT_PROVENANCE_ADDENDUM.md`
- `manifests/RQ1-AUD-01/ARCHIVE_INTEGRITY_REPORT.md`

---

## 1. Executive Summary & Data Contract Architecture

To eliminate training/evaluation discrepancies, data leakage, and silent schema corruptions across Project Langfens, all speech examples ingested by neural acoustic models must adhere to a **strictly typed, version-controlled, immutable data contract**.

This specification defines:
1. The formal schema for standardized speech examples (`LangfensUtteranceRecord`).
2. Exact data structures for token sequences, phone intervals, error events, and label uncertainty flags.
3. Cryptographic provenance fields tracing every example back to raw archive checksums.
4. Formal JSON-Schema and Pydantic validation models for Phase `RQ1-PREP-01`.

```text
Langfens Data Contract Hierarchy
├── 1. Provenance & Archive Ledger (SHA-256, Source Path, Timestamps)
├── 2. Utterance & Demographic Metadata (ID, Speaker, Sex, L1, Corpus)
├── 3. Acoustic References (Sample Rate, Duration, Dynamic Resampling Metadata)
├── 4. Text & Canonical Sequence (Orthography, CMUDict ARPAbet, Stress)
├── 5. Realized Sequence & Error Events (Praat Realization, Sub/Del/Ins Records)
├── 6. Temporal Alignments (Continuous Seconds & Discrete 16kHz Sample Indices)
└── 7. Partition Placeholders (Fold ID, Train/Val/Test Assignment)
```

---

## 2. Formal JSON Record Schema Definition

Below is an annotated exemplar JSON object representing a standardized training/evaluation example from L2-ARCTIC speaker `HQTV` (Vietnamese L1):

```json
{
  "$schema": "https://langfens.org/schemas/v1/utterance_record.schema.json",
  "schema_version": "1.0.0",
  "record_id": "l2arctic_HQTV_arctic_a0001",
  "corpus": "L2-ARCTIC",
  
  "provenance": {
    "raw_archive_name": "l2arctic_release_v5.0.zip",
    "raw_archive_sha256": "490d5a43e48b0af84f6bce8e766ad453e8cd670521d5876bb4d0741cca304fda",
    "source_audio_path": "HQTV/wav/arctic_a0001.wav",
    "source_annotation_path": "HQTV/annotation/arctic_a0001.TextGrid",
    "extraction_timestamp_utc": "2026-10-23T00:00:00Z"
  },

  "speaker_metadata": {
    "speaker_id": "HQTV",
    "biological_sex": "M",
    "native_language_l1": "Vietnamese",
    "target_dialect": "General American English",
    "is_gold_annotated": true
  },

  "audio_metadata": {
    "sample_rate_raw": 44100,
    "sample_rate_target": 16000,
    "channels": 1,
    "bit_depth": 16,
    "duration_seconds": 3.425,
    "total_samples_raw": 151042,
    "total_samples_16k": 54800,
    "signal_peak_amplitude": 0.884,
    "dc_offset": 0.00012,
    "has_clipping": false
  },

  "linguistic_data": {
    "prompt_id": "arctic_a0001",
    "orthographic_transcript": "Author of the danger trail, Philip Stearns, looked down at the street.",
    "canonical_phones_raw": ["AA1", "TH", "ER0", "AH1", "V", "DH", "AH0", "D", "EY1", "N", "JH", "ER0"],
    "canonical_phones_norm39": ["AA", "TH", "ER", "AH", "V", "DH", "AH", "D", "EY", "N", "JH", "ER"],
    "canonical_stress": [1, null, 0, 1, null, null, 0, null, 1, null, null, 0]
  },

  "realized_annotations": {
    "realized_phones_raw": ["AA1", "TH", "AH0", "AH1", "V", "D", "AH0", "D", "EY1", "N", "JH", "AH0"],
    "realized_phones_norm39": ["AA", "TH", "AH", "AH", "V", "D", "AH", "D", "EY", "N", "JH", "AH"],
    "phone_error_rate_per": 16.67
  },

  "phone_intervals": [
    {
      "token_index": 0,
      "canonical_phone": "AA",
      "realized_phone": "AA",
      "canonical_stress": 1,
      "t_start": 0.240,
      "t_end": 0.380,
      "sample_start_16k": 3840,
      "sample_end_16k": 6080,
      "binary_label_strict": 0,
      "binary_label_lenient": 0,
      "error_type": "correct",
      "is_allophonic_accent": false,
      "is_uncertain_err": false,
      "is_typo_correction": false
    },
    {
      "token_index": 2,
      "canonical_phone": "ER",
      "realized_phone": "AH",
      "canonical_stress": 0,
      "t_start": 0.490,
      "t_end": 0.580,
      "sample_start_16k": 7840,
      "sample_end_16k": 9280,
      "binary_label_strict": 1,
      "binary_label_lenient": 1,
      "error_type": "substitution",
      "is_allophonic_accent": false,
      "is_uncertain_err": false,
      "is_typo_correction": false
    },
    {
      "token_index": 5,
      "canonical_phone": "DH",
      "realized_phone": "D",
      "canonical_stress": null,
      "t_start": 0.880,
      "t_end": 0.940,
      "sample_start_16k": 14080,
      "sample_end_16k": 15040,
      "binary_label_strict": 1,
      "binary_label_lenient": 1,
      "error_type": "substitution",
      "is_allophonic_accent": false,
      "is_uncertain_err": false,
      "is_typo_correction": false
    }
  ],

  "error_events": [
    {
      "event_id": "ERR_0001",
      "token_index": 2,
      "error_type": "substitution",
      "canonical_phone": "ER",
      "realized_phone": "AH",
      "articulatory_distance_her": 0.667,
      "is_masked_from_diagnosis": false
    },
    {
      "event_id": "ERR_0002",
      "token_index": 5,
      "error_type": "substitution",
      "canonical_phone": "DH",
      "realized_phone": "D",
      "articulatory_distance_her": 0.333,
      "is_masked_from_diagnosis": false
    }
  ],

  "label_uncertainty_flags": {
    "contains_err_tag": false,
    "contains_asterisk_allophone": false,
    "contains_typo_correction": false,
    "contains_provisional_v8": false,
    "all_tokens_evaluable_39": true
  },

  "split_assignment": {
    "protocol": "4fold_balanced",
    "fold_id": 1,
    "assignment": "train"
  }
}
```

---

## 3. Pydantic Model Data Validation Contract

During implementation phase `RQ1-PREP-01`, all ingested datasets must pass Pydantic validation before being batched into PyTorch tensors:

```python
from pydantic import BaseModel, Field, conint, confloat
from typing import Literal, Optional

class ProvenanceModel(BaseModel):
    raw_archive_name: str
    raw_archive_sha256: str = Field(..., regex=r"^[a-f0-9]{64}$")
    source_audio_path: str
    source_annotation_path: str
    extraction_timestamp_utc: str

class SpeakerMetadataModel(BaseModel):
    speaker_id: str
    biological_sex: Literal["M", "F"]
    native_language_l1: Literal["Arabic", "Chinese", "Hindi", "Korean", "Spanish", "Vietnamese", "Mandarin"]
    target_dialect: str
    is_gold_annotated: bool

class AudioMetadataModel(BaseModel):
    sample_rate_raw: int
    sample_rate_target: int = 16000
    channels: int = 1
    duration_seconds: confloat(gt=0.0)
    total_samples_16k: conint(gt=0)
    has_clipping: bool

class PhoneIntervalModel(BaseModel):
    token_index: Optional[int] = None  # None for insertion events
    canonical_phone: Optional[str] = None
    realized_phone: str
    canonical_stress: Optional[conint(ge=0, le=2)] = None
    t_start: confloat(ge=0.0)
    t_end: confloat(gt=0.0)
    sample_start_16k: int
    sample_end_16k: int
    binary_label_strict: Optional[Literal[0, 1]] = None
    binary_label_lenient: Optional[Literal[0, 1]] = None
    error_type: Literal["correct", "substitution", "deletion", "addition", "silence"]
    is_allophonic_accent: bool
    is_uncertain_err: bool
    is_typo_correction: bool

class ErrorEventModel(BaseModel):
    event_id: str
    token_index: Optional[int]
    error_type: Literal["substitution", "deletion", "addition"]
    canonical_phone: Optional[str]
    realized_phone: str
    is_masked_from_diagnosis: bool

class UncertaintyFlagsModel(BaseModel):
    contains_err_tag: bool
    contains_asterisk_allophone: bool
    contains_typo_correction: bool
    contains_provisional_v8: bool
    all_tokens_evaluable_39: bool

class SplitAssignmentModel(BaseModel):
    protocol: str
    fold_id: int
    assignment: Literal["train", "val", "test"]

class LangfensUtteranceRecord(BaseModel):
    schema_version: str = "1.0.0"
    record_id: str
    corpus: Literal["L2-ARCTIC", "SpeechOcean762"]
    provenance: ProvenanceModel
    speaker_metadata: SpeakerMetadataModel
    audio_metadata: AudioMetadataModel
    phone_intervals: list[PhoneIntervalModel]
    error_events: list[ErrorEventModel]
    label_uncertainty_flags: UncertaintyFlagsModel
    split_assignment: Optional[SplitAssignmentModel] = None
```

---

## 4. Dataloader Collation Contracts for PyTorch Training

During Phase `RQ1-TRN-01`, PyTorch `DataLoader` workers will convert `LangfensUtteranceRecord` items into batch tensors using this collation contract:

```python
import torch

class LangfensBatchCollate:
    """Collates variable-length speech waveforms and phoneme labels into padded tensors."""
    
    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        waveforms = [torch.tensor(item["audio_waveform_16k"]) for item in batch]
        audio_lengths = torch.tensor([w.shape[0] for w in waveforms], dtype=torch.long)
        padded_waveforms = torch.nn.utils.rnn.pad_sequence(waveforms, batch_first=True)
        
        # Binary MDD labels padded with -100 (ignored in cross-entropy loss)
        binary_labels = [torch.tensor(item["binary_labels_strict"]) for item in batch]
        padded_binary_labels = torch.nn.utils.rnn.pad_sequence(
            binary_labels, batch_first=True, padding_value=-100
        )
        
        # Realized CTC targets padded with 0 (<BLANK>)
        ctc_targets = [torch.tensor(item["ctc_target_indices"]) for item in batch]
        target_lengths = torch.tensor([t.shape[0] for t in ctc_targets], dtype=torch.long)
        padded_ctc_targets = torch.nn.utils.rnn.pad_sequence(
            ctc_targets, batch_first=True, padding_value=0
        )
        
        return {
            "net_input": {
                "source": padded_waveforms,
                "padding_mask": (padded_waveforms == 0.0), # or length-based boolean mask
                "audio_lengths": audio_lengths
            },
            "target_binary": padded_binary_labels,
            "target_ctc": padded_ctc_targets,
            "target_lengths": target_lengths,
            "record_ids": [item["record_id"] for item in batch]
        }
```

---

## 5. Storage Formats: Apache Parquet vs. Compressed JSONL

For storing standardized manifests on disk and Google Drive, Project Langfens selects:
- **Primary Manifest Format:** **Apache Parquet (`.parquet`)**
  - High compression ratio ($\approx 85\%$ smaller than raw JSON).
  - Strongly typed column schemas with snappy compression.
  - Fast random chunk reads for distributed dataloader workers.
- **Secondary Inspection Format:** **Line-delimited JSON (`.jsonl`)**
  - Used for developer inspection, debugging, and git-tracked metadata diffs.
- **Location:** `manifests/processed/` (to be created and populated exclusively during `RQ1-PREP-01`).
