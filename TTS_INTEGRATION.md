# TTS Integration Guide for Whisper-WebUI

## Files Created

```
modules/
├── tts/
│   ├── __init__.py
│   ├── tts_base.py        # Base class with SRT→Audio pipeline
│   ├── xtts_inference.py  # XTTS v2 implementation
│   └── data_classes.py    # Gradio UI parameters
└── utils/
    └── paths.py           # Updated with TTS directories
```

## Add to requirements.txt

```
TTS>=0.22.0
librosa
soundfile
scipy
```

## Add TTS Tab to app.py

Insert this after the "BGM Separation" tab:

```python
# Add import at top of app.py
from modules.tts.xtts_inference import XTTSInference
from modules.tts.tts_base import XTTS_LANGUAGES
from modules.utils.paths import TTS_OUTPUT_DIR, TTS_VOICES_DIR

# In App.__init__, add:
self.tts_inf = XTTSInference(
    model_dir=self.args.tts_model_dir if hasattr(self.args, 'tts_model_dir') else TTS_MODELS_DIR,
    output_dir=TTS_OUTPUT_DIR
)

# Add new tab in launch() method:
with gr.TabItem(_("TTS Dubbing")):
    gr.Markdown("### Generate Dubbed Audio from Subtitles")
    
    with gr.Row():
        with gr.Column():
            file_srt = gr.File(
                type="filepath", 
                label=_("Upload Translated SRT File"),
                file_types=[".srt"]
            )
            file_speaker_ref = gr.Audio(
                type="filepath",
                label=_("Voice Reference (10-30 sec of clear speech)"),
                info="Upload audio sample of the voice you want to clone"
            )
        with gr.Column():
            file_original_audio = gr.Audio(
                type="filepath",
                label=_("Original Audio (optional, for mixing)"),
                info="Original video/audio to mix with TTS"
            )
    
    with gr.Row():
        dd_language = gr.Dropdown(
            label=_("Language"),
            choices=list(XTTS_LANGUAGES.keys()),
            value="Polish",
            info="Target language for TTS"
        )
        dd_sync_method = gr.Dropdown(
            label=_("Sync Method"),
            choices=["hybrid", "stretch", "pad"],
            value="hybrid",
            info="How to match TTS to subtitle timing"
        )
    
    with gr.Row():
        cb_mix_original = gr.Checkbox(
            label=_("Mix with Original Audio"),
            value=False
        )
        sl_original_volume = gr.Slider(
            label=_("Original Volume"),
            minimum=0.0,
            maximum=0.5,
            step=0.05,
            value=0.15
        )
    
    with gr.Row():
        btn_generate = gr.Button(_("GENERATE DUBBED AUDIO"), variant="primary")
    
    with gr.Row():
        tb_status = gr.Textbox(label=_("Status"), scale=5)
        file_output = gr.Audio(label=_("Output Audio"), scale=3)
        btn_openfolder = gr.Button('📂', scale=1)
    
    btn_generate.click(
        fn=self.tts_inf.synthesize_from_srt,
        inputs=[
            file_srt,
            file_speaker_ref,
            dd_language,
            gr.Textbox(value="dubbed", visible=False),  # output_name
            file_original_audio,
            cb_mix_original,
            sl_original_volume,
            dd_sync_method
        ],
        outputs=[tb_status, file_output, gr.Files(visible=False)]
    )
    
    btn_openfolder.click(
        fn=lambda: self.open_folder(TTS_OUTPUT_DIR),
        inputs=None,
        outputs=None
    )
```

## Usage Flow

1. **Whisper** → Transcribe video → Russian SRT
2. **Translate** → translate_srt.py → Polish SRT  
3. **TTS Dubbing** → Upload Polish SRT + voice reference → Dubbed audio
4. **FFmpeg** → Mix dubbed audio with original video

## Voice Reference Tips

- 10-30 seconds of clear speech
- Same language as target (Polish)
- No background music/noise
- Single speaker
- Good recording quality

Can use sample from:
- Polish audiobook
- Polish news anchor
- Any Polish speaker recording

## Sync Methods

| Method | Description | Best For |
|--------|-------------|----------|
| `hybrid` | Stretch if 0.7-1.5x, else pad | General use (recommended) |
| `stretch` | Always time-stretch | When timing is critical |
| `pad` | Always pad/trim | Fast generation |
