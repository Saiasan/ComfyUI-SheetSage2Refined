# ComfyUI SheetSage2-Refined

ComfyUI nodes for [SheetSage2-Refined](https://huggingface.co/anime-song/SheetSage2-Refined), a fine-tuned
[SheetSage2](https://huggingface.co/m-a-p/SheetSage2) that transcribes whole songs into beats, downbeats and time
signatures, melody, detailed chords (bass + root x 62 qualities, tensions included) and keys with modes.

Two nodes:

- **Load SheetSage2 Refined Model** loads the base weights from `models/audio_encoders` into ComfyUI's VRAM
  manager. On first run it also downloads `config.json`, the fine-tuned decoder and the frame heads from the
  Hugging Face repo into a `.sheetsage2_refined_folder` next to the weights.
- **SheetSage2 Refined Transcribe** transcribes an `AUDIO` input into chords, keys/beats, structured JSON and an
  ABC score.

## Install

1. Install the model package into ComfyUI's Python:

   ```
   python_embeded\python.exe -m pip install "git+https://github.com/anime-song/SheetSage2-Refined"
   ```

2. Clone this repository into `ComfyUI/custom_nodes`:

   ```
   git clone https://github.com/Saiasan/ComfyUI-SheetSage2Refined ComfyUI/custom_nodes/ComfyUI-SheetSage2-Refined
   ```

3. Download `sheetsage2/model.safetensors` (2.7 GB) from the
   [model repository](https://huggingface.co/anime-song/SheetSage2-Refined/tree/main/sheetsage2) into
   `ComfyUI/models/audio_encoders/`, or into any folder registered as `audio_encoders` in
   `extra_model_paths.yaml`. Rename it as you like (e.g. `SheetSage2-Refined.safetensors`).

4. Restart ComfyUI.

The native `sheetsage2_bf16.safetensors` file from `m-a-p/SheetSage2` is not compatible: the refined package
expects SheetSage2's weights renamed for its own module layout (`sheetsage2/model.safetensors` in the model
repository above).

## Nodes

### Load SheetSage2 Refined Model

| input | |
|---|---|
| `model_name` | the base weights to load, picked from `audio_encoders` |
| `rescoring` | `default` keeps the model's frame-head rescoring, `none` disables it |

### SheetSage2 Refined Transcribe

| input | |
|---|---|
| `sheetsage_model` | output of the loader |
| `audio` | audio to transcribe |
| `key_mode` | write keys with modes (dorian, mixolydian, ...) instead of major/minor |
| `keep_in_vram` | keep the model and its decoder caches loaded between runs |
| `abc_mode` | `melody` (recommended), `full` (melody and chords) or `none` |
| `melody_octave` | transpose the ABC melody by N octaves |

| output | |
|---|---|
| `chords_timeline` | one line per chord: `[start - end] label` |
| `key_and_beats` | key segments, beat and downbeat counts |
| `full_json` | structured annotations: beats, downbeats, chord intervals and labels, keys |
| `abc` | ABC score, ready for the YuE2 Generate Music node |

About the ABC output:

- ComfyUI's ABC writer only notates major and minor keys, so modes are written with the signature of their
  relative major (`D dorian` becomes `K:C`). The modal names stay in `key_and_beats` and `full_json`.
- Chord tensions (`C7(9,13)`) stay in `full_json`; the score writes the base chord (`C7`).
- Melody notes too short to sit on the score's subbeat grid are left out (a warning is logged).
- `melody_octave` only shifts the melody voices; chords, keys and `full_json` are unchanged.

## Notes

- The model runs on the GPU through ComfyUI's model management. With the default batch of 8 windows, expect about
  2.7 GB for the base weights and a few GB for the decoder caches; `keep_in_vram = false` releases both after
  each run.
- The weights are CC BY-NC 4.0 (non-commercial only), like SheetSage2 and MERT-v2. The inference package is MIT.
