"""
Data classes for TTS parameters - Gradio UI integration
"""
from dataclasses import dataclass, fields
from typing import Optional, List
import gradio as gr


@dataclass
class TTSParams:
    """Parameters for TTS synthesis"""
    language: str = "Polish"
    sync_method: str = "hybrid"  # "stretch", "pad", "hybrid"
    mix_original: bool = False
    original_volume: float = 0.15
    temperature: float = 0.7
    repetition_penalty: float = 10.0
    
    @classmethod
    def to_gradio_inputs(cls, defaults: dict = None) -> List[gr.components.Component]:
        """Create Gradio input components"""
        from modules.tts.tts_base import XTTS_LANGUAGES
        
        defaults = defaults or {}
        
        inputs = []
        
        with gr.Row():
            inputs.append(gr.Dropdown(
                label="Language",
                choices=list(XTTS_LANGUAGES.keys()),
                value=defaults.get("language", "Polish"),
                info="Target language for TTS synthesis"
            ))
            inputs.append(gr.Dropdown(
                label="Sync Method",
                choices=["hybrid", "stretch", "pad"],
                value=defaults.get("sync_method", "hybrid"),
                info="How to match TTS duration to subtitle timing. Hybrid recommended."
            ))
        
        with gr.Row():
            inputs.append(gr.Checkbox(
                label="Mix with Original Audio",
                value=defaults.get("mix_original", False),
                info="Keep original audio as background (ambient/music)"
            ))
            inputs.append(gr.Slider(
                label="Original Audio Volume",
                minimum=0.0,
                maximum=0.5,
                step=0.05,
                value=defaults.get("original_volume", 0.15),
                info="Volume of original audio when mixing (0.1-0.2 recommended)"
            ))
        
        with gr.Accordion("Advanced TTS Settings", open=False):
            with gr.Row():
                inputs.append(gr.Slider(
                    label="Temperature",
                    minimum=0.1,
                    maximum=1.0,
                    step=0.1,
                    value=defaults.get("temperature", 0.7),
                    info="Lower = more consistent, higher = more varied"
                ))
                inputs.append(gr.Slider(
                    label="Repetition Penalty",
                    minimum=1.0,
                    maximum=20.0,
                    step=1.0,
                    value=defaults.get("repetition_penalty", 10.0),
                    info="Prevents repetitive speech patterns"
                ))
        
        return inputs


# Language code mappings
XTTS_LANG_CODES = {
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


def get_lang_code(language_name: str) -> str:
    """Convert language name to code"""
    return XTTS_LANG_CODES.get(language_name, "en")
