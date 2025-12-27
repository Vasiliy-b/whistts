import os
import gc
import torch
import gradio as gr
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
from dataclasses import dataclass

from modules.utils.subtitle_manager import read_srt, get_writer
from modules.utils.paths import OUTPUT_DIR


TTS_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "tts")
TTS_MODELS_DIR = os.path.join("models", "TTS")


@dataclass
class TTSSegment:
    """Single TTS segment with timing info"""
    index: int
    start: float  # seconds
    end: float    # seconds
    text: str
    audio: Optional[bytes] = None
    actual_duration: Optional[float] = None


class TTSBase(ABC):
    """Base class for TTS implementations (XTTS, Piper, etc.)"""
    
    def __init__(
        self,
        model_dir: str = TTS_MODELS_DIR,
        output_dir: str = TTS_OUTPUT_DIR
    ):
        super().__init__()
        self.model = None
        self.model_dir = model_dir
        self.output_dir = output_dir
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        self.device = self.get_device()
        self.sample_rate = 24000  # Default, override in subclass
        
    @abstractmethod
    def load_model(self, progress: gr.Progress = gr.Progress()):
        """Load TTS model into memory"""
        pass
    
    @abstractmethod
    def synthesize(
        self,
        text: str,
        speaker_wav: Optional[str] = None,
        language: str = "pl",
        speed: float = 1.0
    ) -> Tuple[bytes, float]:
        """
        Synthesize single text segment
        
        Returns:
            Tuple of (audio_bytes, duration_seconds)
        """
        pass
    
    def synthesize_from_srt(
        self,
        srt_path: str,
        speaker_wav: Optional[str] = None,
        language: str = "pl",
        output_name: str = "dubbed",
        original_audio_path: Optional[str] = None,
        mix_original: bool = False,
        original_volume: float = 0.15,
        sync_method: str = "hybrid",  # "stretch", "pad", "hybrid"
        progress: gr.Progress = gr.Progress()
    ) -> Tuple[str, str, List[str]]:
        """
        Generate dubbed audio from SRT file
        
        Args:
            srt_path: Path to translated SRT file
            speaker_wav: Reference audio for voice cloning (optional)
            language: Target language code
            output_name: Base name for output files
            original_audio_path: Original audio for mixing (optional)
            mix_original: Whether to mix with original audio (background)
            original_volume: Volume of original audio when mixing
            sync_method: How to sync TTS to subtitle timings
            progress: Gradio progress indicator
            
        Returns:
            Tuple of (status_message, output_audio_path, output_files_list)
        """
        import numpy as np
        import soundfile as sf
        from datetime import datetime
        
        try:
            # Load model if not loaded
            if self.model is None:
                self.load_model(progress)
            
            # Parse SRT
            segments = self._parse_srt(srt_path)
            if not segments:
                return "Error: No segments found in SRT", "", []
            
            progress(0.1, desc="Parsed SRT, starting synthesis...")
            
            # Calculate total duration needed
            total_duration = max(seg.end for seg in segments) + 1.0
            
            # Initialize output audio array
            output_audio = np.zeros(int(total_duration * self.sample_rate), dtype=np.float32)
            
            # Synthesize each segment
            for i, seg in enumerate(segments):
                progress((0.1 + 0.7 * i / len(segments)), desc=f"Synthesizing {i+1}/{len(segments)}...")
                
                # Generate TTS audio
                audio_data, actual_duration = self.synthesize(
                    text=seg.text,
                    speaker_wav=speaker_wav,
                    language=language
                )
                
                # Convert bytes to numpy array
                audio_array = np.frombuffer(audio_data, dtype=np.float32)
                
                # Apply time synchronization
                target_duration = seg.end - seg.start
                audio_array = self._sync_audio(
                    audio_array, 
                    actual_duration, 
                    target_duration,
                    method=sync_method
                )
                
                # Place in output at correct position
                start_sample = int(seg.start * self.sample_rate)
                end_sample = start_sample + len(audio_array)
                
                # Handle overlap (fade out existing, fade in new)
                if end_sample <= len(output_audio):
                    # Simple overwrite for now, could add crossfade
                    output_audio[start_sample:end_sample] = audio_array
                else:
                    # Extend if needed
                    output_audio = np.concatenate([
                        output_audio, 
                        np.zeros(end_sample - len(output_audio), dtype=np.float32)
                    ])
                    output_audio[start_sample:end_sample] = audio_array
            
            progress(0.85, desc="Mixing audio...")
            
            # Mix with original if requested
            if mix_original and original_audio_path and os.path.exists(original_audio_path):
                output_audio = self._mix_with_original(
                    output_audio,
                    original_audio_path,
                    original_volume
                )
            
            # Normalize
            max_val = np.abs(output_audio).max()
            if max_val > 0:
                output_audio = output_audio / max_val * 0.95
            
            progress(0.95, desc="Saving output...")
            
            # Save output
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"{output_name}_{timestamp}.wav"
            output_path = os.path.join(self.output_dir, output_filename)
            
            sf.write(output_path, output_audio, self.sample_rate)
            
            status = f"Done! Generated {len(segments)} segments, duration: {total_duration:.1f}s"
            return status, output_path, [output_path]
            
        except Exception as e:
            return f"Error: {str(e)}", "", []
    
    def _parse_srt(self, srt_path: str) -> List[TTSSegment]:
        """Parse SRT file into TTSSegments"""
        import re
        
        segments = []
        
        with open(srt_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split by double newline (segment separator)
        blocks = re.split(r'\n\n+', content.strip())
        
        for block in blocks:
            lines = block.strip().split('\n')
            if len(lines) < 3:
                continue
            
            try:
                index = int(lines[0])
                
                # Parse timecode: 00:00:01,000 --> 00:00:04,500
                time_match = re.match(
                    r'(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[,.](\d{3})',
                    lines[1]
                )
                if not time_match:
                    continue
                
                start = (
                    int(time_match.group(1)) * 3600 +
                    int(time_match.group(2)) * 60 +
                    int(time_match.group(3)) +
                    int(time_match.group(4)) / 1000
                )
                end = (
                    int(time_match.group(5)) * 3600 +
                    int(time_match.group(6)) * 60 +
                    int(time_match.group(7)) +
                    int(time_match.group(8)) / 1000
                )
                
                # Join remaining lines as text
                text = ' '.join(lines[2:]).strip()
                
                if text:
                    segments.append(TTSSegment(
                        index=index,
                        start=start,
                        end=end,
                        text=text
                    ))
            except (ValueError, IndexError):
                continue
        
        return segments
    
    def _sync_audio(
        self,
        audio: 'np.ndarray',
        actual_duration: float,
        target_duration: float,
        method: str = "hybrid"
    ) -> 'np.ndarray':
        """
        Synchronize audio duration to target
        
        Methods:
        - stretch: Time-stretch using librosa/rubberband
        - pad: Pad with silence or trim
        - hybrid: Stretch if ratio is reasonable, else pad/trim
        """
        import numpy as np
        
        if actual_duration <= 0 or target_duration <= 0:
            return audio
        
        ratio = target_duration / actual_duration
        
        if method == "pad" or (method == "hybrid" and (ratio < 0.7 or ratio > 1.5)):
            # Pad or trim
            target_samples = int(target_duration * self.sample_rate)
            if len(audio) > target_samples:
                # Trim with fade out
                audio = audio[:target_samples]
                fade_len = min(int(0.05 * self.sample_rate), len(audio) // 4)
                if fade_len > 0:
                    audio[-fade_len:] *= np.linspace(1, 0, fade_len)
            elif len(audio) < target_samples:
                # Pad with silence
                audio = np.concatenate([
                    audio,
                    np.zeros(target_samples - len(audio), dtype=audio.dtype)
                ])
            return audio
        
        # Time-stretch
        try:
            import librosa
            audio_stretched = librosa.effects.time_stretch(audio, rate=1/ratio)
            return audio_stretched.astype(np.float32)
        except ImportError:
            # Fallback to simple resampling (lower quality)
            from scipy import signal
            target_samples = int(target_duration * self.sample_rate)
            audio_resampled = signal.resample(audio, target_samples)
            return audio_resampled.astype(np.float32)
    
    def _mix_with_original(
        self,
        tts_audio: 'np.ndarray',
        original_path: str,
        original_volume: float = 0.15
    ) -> 'np.ndarray':
        """Mix TTS audio with original (background music/ambience)"""
        import numpy as np
        import soundfile as sf
        from scipy import signal
        
        # Load original
        original, orig_sr = sf.read(original_path)
        
        # Convert to mono if stereo
        if len(original.shape) > 1:
            original = original.mean(axis=1)
        
        # Resample if needed
        if orig_sr != self.sample_rate:
            num_samples = int(len(original) * self.sample_rate / orig_sr)
            original = signal.resample(original, num_samples)
        
        # Match lengths
        if len(original) > len(tts_audio):
            original = original[:len(tts_audio)]
        elif len(original) < len(tts_audio):
            original = np.concatenate([
                original,
                np.zeros(len(tts_audio) - len(original), dtype=original.dtype)
            ])
        
        # Mix
        mixed = tts_audio + original * original_volume
        
        return mixed.astype(np.float32)
    
    @staticmethod
    def get_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch, 'xpu') and torch.xpu.is_available():
            return "xpu"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def offload(self):
        """Offload model and free memory"""
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_max_memory_allocated()
        elif self.device == "xpu":
            torch.xpu.empty_cache()
        
        gc.collect()


# Available languages for XTTS
XTTS_LANGUAGES = {
    "English": "en",
    "Spanish": "es", 
    "French": "fr",
    "German": "de",
    "Italian": "it",
    "Portuguese": "pt",
    "Polish": "pl",
    "Turkish": "tr",
    "Russian": "ru",
    "Dutch": "nl",
    "Czech": "cs",
    "Arabic": "ar",
    "Chinese": "zh-cn",
    "Japanese": "ja",
    "Hungarian": "hu",
    "Korean": "ko",
    "Hindi": "hi",
}
