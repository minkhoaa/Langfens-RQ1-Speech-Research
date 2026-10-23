import io
import os
import tarfile
import zipfile
import tempfile
import shutil
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np
import soundfile as sf
import torch
import torchaudio

@dataclass
class AudioMetrics:
    raw_sr: int
    target_sr: int
    raw_channels: int
    duration_seconds: float
    total_samples_raw: int
    total_samples_target: int
    signal_peak_amplitude: float
    dc_offset: float
    has_clipping: bool

def process_waveform(
    waveform: torch.Tensor,
    normalize_peak: bool = False,
    remove_dc: bool = False
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Applies configurable, optional audio transformations while computing
    signal characteristics. Default is NO modification (preserves source audio).
    """
    signal = waveform.clone()
    dc = signal.mean().item()
    if remove_dc:
        signal = signal - dc
        
    peak = signal.abs().max().item()
    has_clipping = bool(peak >= 0.999)
    
    if normalize_peak and peak > 0.0:
        signal = signal / peak
        
    metrics = {
        "dc_offset": dc,
        "signal_peak_amplitude": peak,
        "has_clipping": has_clipping
    }
    return signal, metrics

class L2ArcticArchiveReader:
    """
    Thread-safe, bounded-memory reader for L2-ARCTIC nested zip archives.
    Caches inner speaker zip handles in memory on-demand to avoid redundant outer zip seeks.
    """
    def __init__(self, master_zip_path: str):
        self.master_zip_path = master_zip_path
        self._master_zf: Optional[zipfile.ZipFile] = None
        self._speaker_zfs: Dict[str, zipfile.ZipFile] = {}

    def _get_master_zf(self) -> zipfile.ZipFile:
        if self._master_zf is None:
            self._master_zf = zipfile.ZipFile(self.master_zip_path, "r")
        return self._master_zf

    def get_speaker_zf(self, speaker: str) -> zipfile.ZipFile:
        if speaker not in self._speaker_zfs:
            master = self._get_master_zf()
            speaker_zip_bytes = master.read(f"{speaker}.zip")
            self._speaker_zfs[speaker] = zipfile.ZipFile(io.BytesIO(speaker_zip_bytes), "r")
        return self._speaker_zfs[speaker]

    def read_wav_bytes(self, speaker: str, utterance_id: str) -> bytes:
        szf = self.get_speaker_zf(speaker)
        member_name = f"{speaker}/wav/{utterance_id}.wav"
        return szf.read(member_name)

    def read_annotation_text(self, speaker: str, utterance_id: str) -> str:
        szf = self.get_speaker_zf(speaker)
        member_name = f"{speaker}/annotation/{utterance_id}.TextGrid"
        return szf.read(member_name).decode("utf-8", errors="replace")

    def close(self):
        for szf in self._speaker_zfs.values():
            szf.close()
        self._speaker_zfs.clear()
        if self._master_zf is not None:
            self._master_zf.close()
            self._master_zf = None

_GLOBAL_L2ARCTIC_READER: Optional[L2ArcticArchiveReader] = None

def get_l2arctic_reader(zip_path: str) -> L2ArcticArchiveReader:
    global _GLOBAL_L2ARCTIC_READER
    if _GLOBAL_L2ARCTIC_READER is None or _GLOBAL_L2ARCTIC_READER.master_zip_path != zip_path:
        _GLOBAL_L2ARCTIC_READER = L2ArcticArchiveReader(zip_path)
    return _GLOBAL_L2ARCTIC_READER

def load_l2arctic_audio_from_zip(
    zip_path: str,
    speaker: str,
    utterance_id: str,
    target_sr: int = 16000,
    normalize_peak: bool = False,
    remove_dc: bool = False
) -> Tuple[torch.Tensor, int, AudioMetrics]:
    """
    Safely reads an L2-ARCTIC WAV file directly from the nested ZIP archive,
    converts to float32 mono tensor, and performs sinc bandlimited resampling.
    """
    reader = get_l2arctic_reader(zip_path)
    wav_bytes = reader.read_wav_bytes(speaker, utterance_id)
    
    audio_np, raw_sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
    raw_channels = 1 if audio_np.ndim == 1 else audio_np.shape[1]
    
    waveform = torch.from_numpy(audio_np)
    if waveform.ndim > 1:
        waveform = waveform.mean(dim=-1)
        
    total_samples_raw = waveform.shape[0]
    duration_seconds = float(total_samples_raw) / float(raw_sr)
    
    if raw_sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, orig_freq=raw_sr, new_freq=target_sr)
        
    waveform, proc_metrics = process_waveform(waveform, normalize_peak=normalize_peak, remove_dc=remove_dc)
    
    metrics = AudioMetrics(
        raw_sr=raw_sr,
        target_sr=target_sr,
        raw_channels=raw_channels,
        duration_seconds=duration_seconds,
        total_samples_raw=total_samples_raw,
        total_samples_target=waveform.shape[0],
        signal_peak_amplitude=proc_metrics["signal_peak_amplitude"],
        dc_offset=proc_metrics["dc_offset"],
        has_clipping=proc_metrics["has_clipping"]
    )
    return waveform, target_sr, metrics

def load_speechocean_audio_from_tar(
    tar_path: str,
    member_path: str,
    target_sr: int = 16000,
    normalize_peak: bool = False,
    remove_dc: bool = False
) -> Tuple[torch.Tensor, int, AudioMetrics]:
    """
    Safely reads a SpeechOcean762 WAV file directly from the tar.gz archive.
    """
    with tarfile.open(tar_path, "r:gz") as tf:
        member = tf.extractfile(member_path)
        if member is None:
            raise FileNotFoundError(f"Member {member_path} not found in {tar_path}")
        wav_bytes = member.read()
        
    audio_np, raw_sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
    raw_channels = 1 if audio_np.ndim == 1 else audio_np.shape[1]
    
    waveform = torch.from_numpy(audio_np)
    if waveform.ndim > 1:
        waveform = waveform.mean(dim=-1)
        
    total_samples_raw = waveform.shape[0]
    duration_seconds = float(total_samples_raw) / float(raw_sr)
    
    if raw_sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, orig_freq=raw_sr, new_freq=target_sr)
        
    waveform, proc_metrics = process_waveform(waveform, normalize_peak=normalize_peak, remove_dc=remove_dc)
    
    metrics = AudioMetrics(
        raw_sr=raw_sr,
        target_sr=target_sr,
        raw_channels=raw_channels,
        duration_seconds=duration_seconds,
        total_samples_raw=total_samples_raw,
        total_samples_target=waveform.shape[0],
        signal_peak_amplitude=proc_metrics["signal_peak_amplitude"],
        dc_offset=proc_metrics["dc_offset"],
        has_clipping=proc_metrics["has_clipping"]
    )
    return waveform, target_sr, metrics

class AudioStagingManager:
    """
    Manages an ephemeral staging directory on local NVMe/SSD to compare
    archive extraction throughput against direct filesystem access.
    """
    def __init__(self, stage_dir: Optional[str] = None):
        if stage_dir is None:
            self._temp_dir = tempfile.mkdtemp(prefix="langfens_audio_stage_")
            self.stage_dir = self._temp_dir
        else:
            self._temp_dir = None
            self.stage_dir = stage_dir
            os.makedirs(self.stage_dir, exist_ok=True)

    def stage_l2arctic_utterances(self, zip_path: str, speaker_utts: Dict[str, list[str]]) -> Dict[str, str]:
        """Extracts requested WAV files into stage_dir and returns mapping utt_id -> local_wav_path."""
        reader = get_l2arctic_reader(zip_path)
        staged_paths: Dict[str, str] = {}
        for spk, utts in speaker_utts.items():
            spk_dir = os.path.join(self.stage_dir, spk)
            os.makedirs(spk_dir, exist_ok=True)
            for utt in utts:
                wav_bytes = reader.read_wav_bytes(spk, utt)
                out_path = os.path.join(spk_dir, f"{utt}.wav")
                with open(out_path, "wb") as f:
                    f.write(wav_bytes)
                staged_paths[utt] = out_path
        return staged_paths

    def cleanup(self):
        if self._temp_dir and os.path.exists(self._temp_dir):
            shutil.rmtree(self._temp_dir)
            self._temp_dir = None
