from typing import List, Dict, Any, Optional
import torch

class LangfensMDDCollate:
    """
    Collates variable-length speech waveforms and phonetic supervision targets
    into padded PyTorch batch tensors conforming to DATA_CONTRACT_SPEC.
    """
    def __init__(self, pad_token_id: int = -100, ctc_blank_id: int = 0):
        self.pad_token_id = pad_token_id
        self.ctc_blank_id = ctc_blank_id

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        waveforms = [item["audio_waveform_16k"] for item in batch]
        audio_lengths = torch.tensor([w.shape[0] for w in waveforms], dtype=torch.long)
        padded_waveforms = torch.nn.utils.rnn.pad_sequence(
            waveforms, batch_first=True, padding_value=0.0
        )
        
        # Boolean padding mask (True where padded)
        max_audio_len = padded_waveforms.shape[1]
        time_indices = torch.arange(max_audio_len).unsqueeze(0)
        padding_mask = time_indices >= audio_lengths.unsqueeze(1)
        
        # Canonical phone prompt tokens (available at train and deploy time)
        if "canonical_phone_indices" in batch[0] and batch[0]["canonical_phone_indices"] is not None:
            can_indices = [item["canonical_phone_indices"] for item in batch]
            canonical_phone_lengths = torch.tensor([c.shape[0] for c in can_indices], dtype=torch.long)
            padded_can_indices = torch.nn.utils.rnn.pad_sequence(
                can_indices, batch_first=True, padding_value=0
            )
        else:
            padded_can_indices = None
            canonical_phone_lengths = None

        # Check if labels are blinded
        if batch[0].get("is_blinded", False) or batch[0]["binary_labels"] is None:
            padded_binary_labels = None
            target_binary_mask = None
            padded_ctc_targets = None
            target_lengths = None
            padded_diagnosis_targets = None
        else:
            binary_labels = [item["binary_labels"] for item in batch]
            padded_binary_labels = torch.nn.utils.rnn.pad_sequence(
                binary_labels, batch_first=True, padding_value=self.pad_token_id
            )
            target_binary_mask = (padded_binary_labels != self.pad_token_id)
            
            ctc_targets = [item["ctc_target_indices"] for item in batch]
            target_lengths = torch.tensor([t.shape[0] for t in ctc_targets], dtype=torch.long)
            padded_ctc_targets = torch.nn.utils.rnn.pad_sequence(
                ctc_targets, batch_first=True, padding_value=self.ctc_blank_id
            )
            
            if "diagnosis_target_indices" in batch[0] and batch[0]["diagnosis_target_indices"] is not None:
                diag_targets = [item["diagnosis_target_indices"] for item in batch]
                padded_diagnosis_targets = torch.nn.utils.rnn.pad_sequence(
                    diag_targets, batch_first=True, padding_value=self.pad_token_id
                )
            else:
                padded_diagnosis_targets = None

            if "aux_phone_target_indices" in batch[0] and batch[0]["aux_phone_target_indices"] is not None:
                aux_targets = [item["aux_phone_target_indices"] for item in batch]
                padded_aux_targets = torch.nn.utils.rnn.pad_sequence(
                    aux_targets, batch_first=True, padding_value=self.pad_token_id
                )
            else:
                padded_aux_targets = None
            
        return {
            "net_input": {
                "source": padded_waveforms,
                "padding_mask": padding_mask,
                "audio_lengths": audio_lengths,
                "canonical_phone_indices": padded_can_indices,
                "canonical_phone_lengths": canonical_phone_lengths
            },
            "canonical_phone_indices": padded_can_indices,
            "canonical_phone_lengths": canonical_phone_lengths,
            "target_binary": padded_binary_labels,
            "target_binary_mask": target_binary_mask,
            "target_ctc": padded_ctc_targets,
            "target_lengths": target_lengths,
            "target_diagnosis": padded_diagnosis_targets,
            "target_aux_phone": padded_aux_targets,
            "record_ids": [item["record_id"] for item in batch]
        }
