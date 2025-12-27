import os
import io
import torch
import gradio as gr
import numpy as np
from typing import Tuple, Optional, List

from modules.tts.tts_base import TTSBase, TTS_MODELS_DIR, XTTS_LANGUAGES
from modules.utils.logger import get_logger

logger = get_logger()


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
        
    def load_model(self, progress: gr.Progress = gr.Progress()):
        """Load XTTS v2 model"""
        if self.model is not None:
            logger.info("XTTS model already loaded")
            return
        
        progress(0.1, desc="Loading XTTS v2 model...")
        
        try:
            from TTS.tts.configs.xtts_config import XttsConfig
            from TTS.tts.models.xtts import Xtts
            
            # Model path
            model_path = os.path.join(self.model_dir, "xtts_v2")
            
            # Download if not exists
            if not os.path.exists(model_path):
                progress(0.2, desc="Downloading XTTS v2 model (~1.8GB)...")
                os.makedirs(model_path, exist_ok=True)
                
                # Use TTS library's download mechanism
                from TTS.utils.manage import ModelManager
                manager = ModelManager()
                model_path, config_path, _ = manager.download_model("tts_models/multilingual/multi-dataset/xtts_v2")
                
                logger.info(f"XTTS model downloaded to: {model_path}")
            
            progress(0.5, desc="Loading model weights...")
            
            # Load config
            config_path = os.path.join(model_path, "config.json")
            if os.path.exists(config_path):
                self.config = XttsConfig()
                self.config.load_json(config_path)
            else:
                # Use default config
                self.config = XttsConfig()
            
            # Load model
            self.model = Xtts.init_from_config(self.config)
            
            checkpoint_path = os.path.join(model_path, "model.pth")
            vocab_path = os.path.join(model_path, "vocab.json")
            
            if os.path.exists(checkpoint_path):
                self.model.load_checkpoint(
                    self.config,
                    checkpoint_path=checkpoint_path,
                    vocab_path=vocab_path if os.path.exists(vocab_path) else None,
                    use_deepspeed=False
                )
            else:
                # Alternative: load from TTS library
                from TTS.api import TTS
                tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
                self.model = tts.synthesizer.tts_model
                self.config = tts.synthesizer.tts_config
            
            self.model = self.model.to(self.device)
            self.model.eval()
            
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
        speed: float = 1.0
    ) -> Tuple[bytes, float]:
        """
        Synthesize speech from text
        
        Args:
            text: Text to synthesize
            speaker_wav: Path to reference audio for voice cloning
            language: Language code (e.g., "pl" for Polish)
            speed: Speech speed multiplier (not directly supported, use sync)
            
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
                    temperature=0.7,
                    length_penalty=1.0,
                    repetition_penalty=10.0,
                    top_k=50,
                    top_p=0.85,
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
        # Create a simple default voice by using model's default
        # This is a workaround - ideally user should provide reference
        
        if self.default_speaker_wav and os.path.exists(self.default_speaker_wav):
            return self.model.get_conditioning_latents(
                audio_path=[self.default_speaker_wav]
            )
        
        # Generate dummy latents (model will use internal defaults)
        # This is not ideal but works as fallback
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
