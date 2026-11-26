import json
import logging
import os
import re
import shutil
import tempfile
import wave
import numpy as np
import torch
from huggingface_hub import hf_hub_download

import comfy.model_management as mm
import comfy.model_patcher
import folder_paths
from comfy.audio_encoders.sheetsage2_abc import MelodyVoiceError, events_to_abc

from sheetsage2_refined import SheetSage2Refined, events_to_annotations
from sheetsage2_refined.theory import PITCH_CLASS, SHARP_NAMES


def _link_or_copy(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst) or os.path.islink(dst):
        os.remove(dst)
    try:
        os.link(src, dst)
    except OSError:
        try:
            os.symlink(src, dst)
        except OSError:
            shutil.copy2(src, dst)


def _extract_repo_files(obj, found=None):
    """Walk config.json recursively to find every file path it names."""
    if found is None:
        found = set()
    if isinstance(obj, str):
        if obj.endswith((".safetensors", ".pt", ".bin", ".json")):
            found.add(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            _extract_repo_files(v, found)
    elif isinstance(obj, list):
        for item in obj:
            _extract_repo_files(item, found)
    return found


def _save_wav_stdlib(wav_path, waveform, sample_rate):
    """Write a 16-bit PCM WAV with the standard library wave module (no torchcodec)."""
    wav_np = waveform.detach().cpu().float().numpy()
    if wav_np.ndim == 1:
        wav_np = wav_np[None, :]  # [1, samples]
    num_channels = int(wav_np.shape[0])
    wav_int16 = (np.clip(wav_np, -1.0, 1.0) * 32767.0).astype(np.int16)
    interleaved = wav_int16.T.tobytes()

    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(num_channels)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(int(sample_rate))
        wf.writeframes(interleaved)


def _jsonable(value):
    """Turn the numpy arrays of events_to_annotations into JSON lists."""
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


# Mode -> semitones between its tonic and the tonic of the major key carrying its signature
_MODE_STEPS = {"major": 0, "dorian": 2, "phrygian": 4, "lydian": 5, "mixolydian": 7, "minor": 9, "locrian": 11}


def _abc_key(key):
    """ComfyUI's ABC writer only notates major/minor: a mode becomes the signature of its relative major."""
    if key == "N":
        return "C:major"
    name, _, mode = key.partition(":")
    tonic = PITCH_CLASS.get(name)
    if mode in ("major", "minor") or tonic is None or mode not in _MODE_STEPS:
        return key
    return SHARP_NAMES[(tonic - _MODE_STEPS[mode]) % 12] + ":major"


def _abc_events(events, octaves=0):
    """Copy the events with the keys and chords ComfyUI's ABC writer can notate (without tensions)."""
    out = []
    for event in events:
        values = dict(event["values"])
        if "melody" in values:
            values["melody"] = [dict(note, pitch=note["pitch"] + 12 * octaves) for note in values["melody"]]
        if "key" in values:
            values["key"] = _abc_key(values["key"])
        chord = values.get("chord", "")
        if ":" in chord:
            root, _, descriptor = chord.partition(":")
            quality, slash, bass = descriptor.partition("/")
            quality = quality.partition("(")[0]  # tensions stay in the JSON output
            values["chord"] = f"{root}:{'maj' if quality == '5' else quality}" + (f"/{bass}" if slash else "")
        out.append({**event, "values": values})
    return out


_UNPLACEABLE_NOTE_RE = re.compile(r"note pitch=(\d+) at ([-\d.]+)-([-\d.]+) cannot be represented")


def _drop_melody_note(events, start, end, pitch):
    for event in events:
        if abs(event["time"] - start) > 1e-6:
            continue
        notes = event["values"].get("melody")
        for index, note in enumerate(notes or ()):
            if note["pitch"] == pitch and abs(note["end_time"] - end) <= 1e-6:
                del notes[index]
                return True
    return False


def _abc_score(events, duration, melody_only, octaves=0):
    """ABC score of the events; drop the notes too short for the ABC writer's subbeat grid."""
    events = _abc_events(events, octaves)
    dropped, error = 0, None
    for _ in range(512):
        try:
            abc = events_to_abc(events, duration, melody_only=melody_only)
        except MelodyVoiceError as exc:
            error = exc
            match = _UNPLACEABLE_NOTE_RE.search(str(exc))
            if match is None or not _drop_melody_note(events, float(match[2]), float(match[3]), int(match[1])):
                raise
            dropped += 1
            continue
        if dropped:
            logging.warning("SheetSage2 Refined: %s melody notes too short for the ABC grid were left out of the score.", dropped)
        return abc
    raise error


class SheetSage2RefinedModule(torch.nn.Module):
    """Every torch module SheetSage2Refined runs, gathered for ComfyUI's ModelPatcher.

    The wrapper keeps its rescoring heads outside ``model``: patching ``model`` alone moves the
    encoder to the GPU but leaves those heads, which read its states, on the CPU.
    """

    def __init__(self, wrapper):
        super().__init__()
        self.wrapper = wrapper
        self.model = wrapper.model
        # heads already inside ``model`` (bar_features) must not be registered twice
        in_model = {id(parameter) for parameter in self.model.parameters()}
        heads = {}
        for head, _ in wrapper.rescoring.values():
            heads[id(head)] = head
        self.heads = torch.nn.ModuleList(
            head
            for head in heads.values()
            if not all(id(parameter) in in_model for parameter in head.parameters())
        )


class SheetSage2RefinedLoader:
    """Load a local .safetensors file into ComfyUI's VRAM manager."""

    @classmethod
    def INPUT_TYPES(cls):
        files = [
            f for f in folder_paths.get_filename_list("audio_encoders")
            if f.endswith(".safetensors") and not f.startswith(".")
        ]
        if not files:
            files = ["No .safetensors file found in models/audio_encoders"]

        return {
            "required": {
                "model_name": (files,),
            },
            "optional": {
                "rescoring": (["default", "none"], {"default": "default"}),
            }
        }

    RETURN_TYPES = ("SHEETSAGE2_REFINED_MODEL",)
    RETURN_NAMES = ("sheetsage_model",)
    FUNCTION = "load_model"
    CATEGORY = "audio/loaders"

    def load_model(self, model_name, rescoring="default"):
        # ComfyUI runs nodes in inference mode: load outside it to keep normal tensors
        with torch.inference_mode(False):
            model_path = folder_paths.get_full_path("audio_encoders", model_name)
            if not model_path or not os.path.exists(model_path):
                raise FileNotFoundError(f"File not found: {model_path}")

            load_device = mm.get_torch_device()
            offload_device = mm.unet_offload_device()

            base_dir = os.path.dirname(model_path)
            local_repo_dir = os.path.join(base_dir, ".sheetsage2_refined_folder")
            os.makedirs(local_repo_dir, exist_ok=True)

            # 1. Fetch config.json if it is missing
            config_path = os.path.join(local_repo_dir, "config.json")
            if not os.path.exists(config_path):
                hf_hub_download(
                    repo_id="anime-song/SheetSage2-Refined",
                    filename="config.json",
                    local_dir=local_repo_dir,
                )

            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)

            # 2. Link the local .safetensors to the exact path config["base_weights"] expects
            base_weights_rel = config.get("base_weights", "sheetsage2/model.safetensors")
            target_st = os.path.join(local_repo_dir, *base_weights_rel.split("/"))
            _link_or_copy(model_path, target_st)

            # 3. Download only the small auxiliary files that are missing
            all_files = _extract_repo_files(config)
            for rel_file in sorted(all_files):
                if rel_file == base_weights_rel:
                    continue
                local_file_path = os.path.join(local_repo_dir, *rel_file.split("/"))
                if not os.path.exists(local_file_path):
                    hf_hub_download(
                        repo_id="anime-song/SheetSage2-Refined",
                        filename=rel_file,
                        local_dir=local_repo_dir,
                    )

            # 4. Load through the official API on CPU: the ModelPatcher moves it to the GPU
            rescoring_arg = None if rescoring == "default" else []
            wrapper = SheetSage2Refined.load(
                folder=local_repo_dir,
                device="cpu",
                rescoring=rescoring_arg,
            )

            patcher = comfy.model_patcher.ModelPatcher(
                SheetSage2RefinedModule(wrapper),
                load_device=load_device,
                offload_device=offload_device,
            )
            return (patcher,)


class SheetSage2RefinedTranscribe:
    """Transcribe audio into chords, keys, beats, structured JSON and an ABC score."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sheetsage_model": ("SHEETSAGE2_REFINED_MODEL",),
                "audio": ("AUDIO",),
                "key_mode": ("BOOLEAN", {"default": True}),
                "keep_in_vram": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "abc_mode": (["melody", "full", "none"], {
                    "default": "melody",
                    "tooltip": "ABC score from the same transcription: melody (melody only, recommended) "
                               "or full (melody and chords). none: no score.",
                }),
                "melody_octave": ("INT", {
                    "default": 0, "min": -2, "max": 2, "step": 1,
                    "tooltip": "Transpose the ABC melody voice by N octaves (chords, keys and JSON are unchanged).",
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("chords_timeline", "key_and_beats", "full_json", "abc")
    FUNCTION = "transcribe"
    CATEGORY = "audio/transcription"

    def transcribe(self, sheetsage_model, audio, key_mode, keep_in_vram, abc_mode="melody", melody_octave=0):
        with torch.inference_mode(False):
            mm.load_model_gpu(sheetsage_model)
            wrapper = sheetsage_model.model.wrapper

            try:
                waveform = audio["waveform"]
                if waveform.dim() == 3:
                    waveform = waveform.squeeze(0)
                sample_rate = audio["sample_rate"]
                duration_seconds = waveform.shape[-1] / float(sample_rate)

                # key_mode swaps the "key" prompt for "key_mode", like the package's --key-mode option
                prompts = None
                if key_mode:
                    prompts = tuple("key_mode" if prompt == "key" else prompt for prompt in wrapper.prompts)

                with tempfile.TemporaryDirectory() as tmpdir:
                    wav_path = os.path.join(tmpdir, "input.wav")
                    _save_wav_stdlib(wav_path, waveform, sample_rate)
                    events = wrapper.transcribe([wav_path], prompts=prompts)[0]

                annotations = events_to_annotations(events, duration_seconds)

                chords_lines = [
                    f"[{start:.2f}s - {end:.2f}s] {label}"
                    for (start, end), label in zip(
                        annotations.get("chord_intervals", []),
                        annotations.get("chord_labels", []),
                    )
                ]
                chords_timeline = "\n".join(chords_lines)

                keys = ", ".join(str(k) for k in annotations.get("keys", [])) or "Unknown"
                beats_count = len(annotations.get("beats", []))
                downbeats_count = len(annotations.get("downbeats", []))
                key_and_beats = f"Keys: {keys} | Beats: {beats_count} | Downbeats: {downbeats_count}"

                full_json = json.dumps(_jsonable(annotations), indent=2, ensure_ascii=False)

                abc = ""
                if abc_mode != "none":
                    abc = _abc_score(events, duration_seconds, melody_only=abc_mode == "melody", octaves=melody_octave)

                return (chords_timeline, key_and_beats, full_json, abc)

            finally:
                if not keep_in_vram:
                    # Also release the package's BatchDecoder (KV caches and CUDA graphs) before the offload
                    wrapper.model._batch_decoder = None
                    sheetsage_model.detach()
                    mm.soft_empty_cache()


NODE_CLASS_MAPPINGS = {
    "SheetSage2RefinedLoader": SheetSage2RefinedLoader,
    "SheetSage2RefinedTranscribe": SheetSage2RefinedTranscribe,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SheetSage2RefinedLoader": "Load SheetSage2 Refined Model",
    "SheetSage2RefinedTranscribe": "SheetSage2 Refined Transcribe",
}
