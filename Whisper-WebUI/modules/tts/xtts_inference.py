# Patch torch._dynamo before any imports (required for torch 2.2.0 + transformers 4.44.2)
import sys
import os

class _FakeDynamo:
    @staticmethod
    def is_compiling():
        return False

if 'torch._dynamo' not in sys.modules:
    sys.modules['torch._dynamo'] = _FakeDynamo()

# Auto-agree to Coqui TOS for non-interactive usage
os.environ.setdefault('COQUI_TOS_AGREED', '1')

import torch
import gradio as gr
import numpy as np
from typing import Tuple, Optional, List

from modules.tts.tts_base import TTSBase, TTS_MODELS_DIR, XTTS_LANGUAGES
from modules.utils.logger import get_logger

logger = get_logger()


# Optimized inference parameters (tested for Polish lector-style dubbing)
DEFAULT_TTS_PARAMS = {
    'temperature': 0.45,      # Lower = more stable/consistent
    'top_p': 0.85,            # Nucleus sampling threshold
    'top_k': 35,              # Limits vocabulary choices
    'repetition_penalty': 15.0,  # Prevents repetitive patterns
    'length_penalty': 1.1,    # Slightly longer pauses
}


class XTTSInference(TTSBase):
    """
    XTTS v2 (Coqui TTS) implementation
    
    Features:
    - Voice cloning from reference audio
    - Multi-language support including Polish
    - Good quality for "lector" style dubbing
    
    VRAM: ~6GB
    """
    
    def __init__(
        self,
        model_dir: str = TTS_MODELS_DIR,
        output_dir: str = None
    ):
        super().__init__(model_dir=model_dir, output_dir=output_dir)
        self.model = None
        self.config = None
        self.sample_rate = 24000  # XTTS default
        self.available_languages = list(XTTS_LANGUAGES.keys())
        self.default_speaker_wav = None
        self.tts_params = DEFAULT_TTS_PARAMS.copy()
        
    def load_model(self, progress: gr.Progress = gr.Progress()):
        """Load XTTS v2 model"""
        if self.model is not None:
            logger.info("XTTS model already loaded")
            return
        
        progress(0.1, desc="Loading XTTS v2 model...")
        
        try:
            # Try direct model loading first (faster if already downloaded)
            from TTS.api import TTS
            
            progress(0.3, desc="Initializing XTTS v2...")
            tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
            
            progress(0.7, desc="Moving to GPU...")
            tts = tts.to(self.device)
            
            self.model = tts.synthesizer.tts_model
            self.config = tts.synthesizer.tts_config
            self._tts_api = tts  # Keep reference for simple API
            
            progress(1.0, desc="XTTS model loaded!")
            logger.info(f"XTTS v2 loaded on {self.device}")
            
        except ImportError as e:
            raise ImportError(
                "TTS library not installed. Run: pip install TTS>=0.22.0"
            ) from e
        except Exception as e:
            logger.error(f"Error loading XTTS: {e}")
            raise
    
    def synthesize(
        self,
        text: str,
        speaker_wav: Optional[str] = None,
        language: str = "pl",
        speed: float = 1.0,
        **kwargs
    ) -> Tuple[bytes, float]:
        """
        Synthesize speech from text
        
        Args:
            text: Text to synthesize
            speaker_wav: Path to reference audio for voice cloning
            language: Language code (e.g., "pl" for Polish)
            speed: Speech speed multiplier (not directly supported, use sync)
            **kwargs: Override default TTS params (temperature, top_p, etc.)
            
        Returns:
            Tuple of (audio_bytes as float32, duration_seconds)
        """
        if self.model is None:
            self.load_model()
        
        # Resolve language code
        if language in XTTS_LANGUAGES:
            language = XTTS_LANGUAGES[language]
        
        # Clean text
        text = self._clean_text(text)
        
        if not text.strip():
            # Return silence for empty text
            silence = np.zeros(int(0.5 * self.sample_rate), dtype=np.float32)
            return silence.tobytes(), 0.5
        
        # Merge params
        params = {**self.tts_params, **kwargs}
        
        try:
            # Get speaker embedding
            if speaker_wav and os.path.exists(speaker_wav):
                gpt_cond_latent, speaker_embedding = self.model.get_conditioning_latents(
                    audio_path=[speaker_wav]
                )
            else:
                # Use default/random speaker
                gpt_cond_latent, speaker_embedding = self._get_default_speaker()
            
            # Synthesize
            with torch.no_grad():
                outputs = self.model.inference(
                    text=text,
                    language=language,
                    gpt_cond_latent=gpt_cond_latent,
                    speaker_embedding=speaker_embedding,
                    temperature=params['temperature'],
                    top_p=params['top_p'],
                    top_k=params['top_k'],
                    repetition_penalty=params['repetition_penalty'],
                    length_penalty=params['length_penalty'],
                    enable_text_splitting=True
                )
            
            # Get audio
            audio = outputs["wav"]
            
            # Convert to numpy if tensor
            if isinstance(audio, torch.Tensor):
                audio = audio.cpu().numpy()
            
            # Ensure float32
            audio = audio.astype(np.float32)
            
            # Calculate duration
            duration = len(audio) / self.sample_rate
            
            return audio.tobytes(), duration
            
        except Exception as e:
            logger.error(f"XTTS synthesis error: {e}")
            # Return silence on error
            silence = np.zeros(int(1.0 * self.sample_rate), dtype=np.float32)
            return silence.tobytes(), 1.0
    
    def _get_default_speaker(self):
        """Get default speaker embedding when no reference provided"""
        if self.default_speaker_wav and os.path.exists(self.default_speaker_wav):
            return self.model.get_conditioning_latents(
                audio_path=[self.default_speaker_wav]
            )
        
        # Generate dummy latents (model will use internal defaults)
        gpt_cond_latent = torch.zeros(1, 1024).to(self.device)
        speaker_embedding = torch.zeros(1, 512).to(self.device)
        
        return gpt_cond_latent, speaker_embedding
    
    def _clean_text(self, text: str) -> str:
        """Clean text for TTS"""
        import re
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Remove subtitle artifacts
        text = re.sub(r'\[.*?\]', '', text)  # [music], [laughter], etc.
        text = re.sub(r'\(.*?\)', '', text)  # (inaudible), etc.
        
        # Fix common issues
        text = text.replace('...', '.')
        text = text.replace('..', '.')
        
        return text.strip()
    
    def set_default_speaker(self, speaker_wav: str):
        """Set default speaker for when no reference is provided"""
        if os.path.exists(speaker_wav):
            self.default_speaker_wav = speaker_wav
            logger.info(f"Default speaker set to: {speaker_wav}")
    
    def set_params(self, **kwargs):
        """Update TTS parameters"""
        for key in kwargs:
            if key in self.tts_params:
                self.tts_params[key] = kwargs[key]
                logger.info(f"TTS param {key} set to {kwargs[key]}")
    
    def get_available_voices(self) -> List[str]:
        """Get list of available voice presets (if any cached)"""
        voices_dir = os.path.join(self.model_dir, "voices")
        if os.path.exists(voices_dir):
            return [f for f in os.listdir(voices_dir) if f.endswith(('.wav', '.mp3'))]
        return []


class XTTSSimple:
    """
    Simplified XTTS wrapper using TTS library directly
    Easier to use but less control
    """
    
    def __init__(self):
        self.tts = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sample_rate = 24000
        
    def load(self):
        """Load model using TTS library"""
        from TTS.api import TTS
        self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(self.device)
        
    def synthesize_to_file(
        self,
        text: str,
        output_path: str,
        speaker_wav: str,
        language: str = "pl"
    ):
        """Synthesize directly to file"""
        if self.tts is None:
            self.load()
        
        self.tts.tts_to_file(
            text=text,
            file_path=output_path,
            speaker_wav=speaker_wav,
            language=language
        )
        
    def synthesize(
        self,
        text: str,
        speaker_wav: str,
        language: str = "pl"
    ) -> np.ndarray:
        """Synthesize and return numpy array"""
        if self.tts is None:
            self.load()
        
        audio = self.tts.tts(
            text=text,
            speaker_wav=speaker_wav,
            language=language
        )
        
        return np.array(audio, dtype=np.float32)
