"""NISQA speech quality and MuQ music-reference affinity."""

import inspect
import json
import sys

import librosa
import numpy as np
import torch
import torch.nn.functional as F


class Nisqa:
    def __init__(self, device, assets):
        self.device = device
        self.assets = assets
        torch.set_num_threads(2)
        # The checkpoint carries both configuration and model state.
        sys.path.insert(0, str(assets["nisqa_source"]))
        from nisqa import NISQA_lib as NL

        self.NL = NL
        ck = torch.load(assets["nisqa"], map_location="cpu", weights_only=False)
        self.args = ck["args"]
        assert self.args["model"] == "NISQA_DIM"
        params = {k: self.args[k] for k in inspect.signature(NL.NISQA_DIM).parameters if k in self.args}
        self.model = NL.NISQA_DIM(**params)
        self.model.load_state_dict(ck["model_state_dict"])
        self.model.to(self.device).eval()
        print(
            "nisqa_loaded",
            json.dumps({k: v for k, v in self.args.items() if k.startswith("ms_")}, default=str),
            flush=True,
        )

    @torch.inference_mode()
    def score(self, y, sr):
        p = self.args
        s = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_fft=p["ms_n_fft"],
            hop_length=int(sr * p["ms_hop_length"]),
            win_length=int(sr * p["ms_win_length"]),
            window="hann",
            center=True,
            pad_mode="reflect",
            power=1.0,
            n_mels=p["ms_n_mels"],
            fmin=0.0,
            fmax=p["ms_fmax"],
            htk=False,
            norm="slaney",
        )
        s = librosa.amplitude_to_db(s, ref=1.0, amin=1e-4, top_db=80.0)
        x, n = self.NL.segment_specs(
            "decoded_memory", s, p["ms_seg_length"], p["ms_seg_hop_length"], p["ms_max_segments"]
        )
        z = (
            self.model(x.float()[None].to(self.device), torch.tensor([int(n)], device=self.device))
            .flatten()
            .cpu()
            .numpy()
        )
        assert len(z) == 5 and np.isfinite(z).all()
        return dict(zip(["mos", "noisiness", "discontinuity", "coloration", "loudness"], map(float, z)))


class Music:
    def __init__(self, device, assets):
        from muq import MuQ

        self.device = device
        self.assets = assets
        torch.set_num_threads(2)
        torch.manual_seed(0)
        self.model = MuQ.from_pretrained(str(assets["muq"])).to(self.device).eval()
        print("muq_loaded", flush=True)

    @torch.inference_mode()
    def encode(self, x):
        values = []
        weights = []
        for start in range(0, len(x), 240000):
            clip = x[start : start + 240000]
            weights.append(len(clip))
            clip = np.pad(clip, (0, max(0, 24000 - len(clip))))
            out = self.model(torch.from_numpy(clip.copy())[None].to(self.device), output_hidden_states=True)
            h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out.hidden_states[-1]
            values.append(F.normalize(h.mean(dim=1).float(), dim=-1)[0].cpu().numpy())
        z = np.average(np.stack(values), axis=0, weights=weights)
        z = z / max(np.linalg.norm(z), 1e-12)
        onset = librosa.onset.onset_strength(y=x, sr=24000)
        tempo, beats = librosa.beat.beat_track(onset_envelope=onset, sr=24000)
        chroma = librosa.feature.chroma_stft(y=x, sr=24000, n_fft=2048, hop_length=512)
        # Normalize each chroma frame; similarity has no uniform task-quality direction.
        chroma /= np.maximum(np.linalg.norm(chroma, axis=0, keepdims=True), 1e-8)
        return z, {
            "tempo_bpm": float(np.asarray(tempo).reshape(-1)[0]),
            "beat_times_s": librosa.frames_to_time(beats, sr=24000).tolist(),
            "chroma_mean": chroma.mean(axis=1).tolist(),
            "chroma_frames": chroma.tolist(),
        }
