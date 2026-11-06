# Third-party dependencies and assets

The Apache-2.0 license covers original ORAV source in this repository. It does not
relicense external packages, model weights, media, or API services. No third-party
model implementation, checkpoint, FFmpeg binary, or benchmark media is bundled.
Adapters import separately installed implementations or explicitly supplied source
checkouts. An adapter being optional does not remove its dependency's license
conditions.

## Core runtime

Python, NumPy, Pillow, and jsonschema retain their respective licenses. Preserve
their notices when redistributing an environment. FFmpeg is invoked as a separate
program. Its license depends on build options: GPL-enabled builds must be handled
under the applicable GPL terms; see [FFmpeg licensing](https://ffmpeg.org/legal.html).
Check the actual binary before distributing it or a container containing it.

The reasoning judge and listener are external services. Users supply their own
endpoints and credentials and remain subject to those services' terms.

## Diagnostic components requiring particular attention

| Component | Upstream terms / release boundary |
|---|---|
| [DOVER / DOVER++](https://github.com/VQAssessment/DOVER/blob/master/LICENSE) | S-Lab License 1.0 restricts the licensed implementation to noncommercial use absent additional permission. The adapter does not redistribute it. |
| [CoTracker3](https://github.com/facebookresearch/co-tracker/blob/main/LICENSE.md) | Majority CC-BY-NC 4.0; some portions have separate licenses. Noncommercial conditions remain applicable. |
| [NISQA](https://github.com/gabrielmittag/NISQA#paper-and-license) | Code MIT; pretrained weights CC-BY-NC-SA 4.0. Code and weights have different permissions. |
| [Aesthetic Predictor V2.5 implementation](https://github.com/discus0434/aesthetic-predictor-v2-5/blob/main/LICENSE) | Upstream currently identifies AGPL-3.0. Confirm the exact checkout's license and the obligations for any combined/distributed application or service. The head and SigLIP backbone require separate checks. |
| [Audiobox Aesthetics](https://github.com/facebookresearch/audiobox-aesthetics#license) | Majority CC-BY 4.0, with separately licensed MIT portions. Preserve attribution and applicable notices. There is no NC restriction in CC-BY 4.0. |
| [ArcFace implementation](https://github.com/ronghuaiyang/arcface-pytorch) | This source and the ResNet18 checkpoint need an explicit permission check; do not infer permission from the paper's availability or from another ArcFace implementation's license. |
| [ALADIN](https://github.com/DanRuta/ALADIN) | Confirm the exact source/checkpoint terms before using or redistributing them. This repository supplies only the calling adapter. |

## Other diagnostic dependencies

Additional upstream sources and model downloads: [CLIP](https://github.com/openai/CLIP), [DINO](https://github.com/facebookresearch/dino), [VBench](https://github.com/Vchitect/VBench), [OpenCLIP](https://github.com/mlfoundations/open_clip), [PyIQA/MUSIQ](https://github.com/chaofengc/IQA-PyTorch), [AMT](https://github.com/MCG-NKU/AMT), [LAION aesthetic predictor](https://github.com/LAION-AI/aesthetic-predictor), [SigLIP](https://huggingface.co/google/siglip-so400m-patch14-384), [SpeechBrain/ECAPA](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb), [Silero VAD](https://github.com/snakers4/silero-vad), [Whisper](https://huggingface.co/openai/whisper-small), [DNSMOS](https://github.com/microsoft/DNS-Challenge), [CLAP](https://huggingface.co/laion/clap-htsat-unfused), [MuQ](https://huggingface.co/OpenMuQ/MuQ-large-msd-iter), [RetinaFace](https://github.com/ternaus/retinaface), [GroundingDINO](https://huggingface.co/IDEA-Research/grounding-dino-tiny), and [SAM2.1](https://huggingface.co/facebook/sam2.1-hiera-tiny).

The code also uses Transformers, PyTorch, torchvision, torchaudio, ONNX Runtime, SciPy, librosa, OpenCC, and supporting Python libraries.
Their code and checkpoint licenses must be retained independently. A top-level
project license is not evidence that every model asset has identical terms.

Official code licenses include MIT for
[CLIP](https://github.com/openai/CLIP/blob/main/LICENSE) and
[Whisper](https://github.com/openai/whisper/blob/main/LICENSE), and Apache-2.0 for
[DINO](https://github.com/facebookresearch/dino/blob/main/LICENSE) and
[VBench](https://github.com/Vchitect/VBench/blob/master/LICENSE).
Diagnostic runs record the assets and package versions actually used. Before
distributing a complete environment, review the terms of those components.

Acknowledging a component in a paper does not replace retaining the copyright and
license notices required by its terms. Benchmark media and generated outputs are
outside this repository's code license.
