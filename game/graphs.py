#!/usr/bin/env python
"""
graphs.py -- the two ComfyUI graphs behind POV Switch, as API-format prompts.

  pov_frame(...)     Qwen-Image-Edit-2509 (GGUF Q4 + 4-step Lightning LoRA): turn the source frame
                     into the chosen character's first-person view. One image in, one image out.
  outcome_clip(...)  Wan2.2 I2V A14B (GGUF Q4 high + low noise experts, 4-step lightx2v LoRAs):
                     play the chosen outcome forward from that frame. One image in, one mp4 out.

Both name their input files by path; `comfy deploy run --asset-root <dir>` uploads them.
"""
from __future__ import annotations

from pathlib import Path

MODELS = dict(
    qwen_unet="qwen-image-edit-2509-Q4_K_M.gguf",
    qwen_clip="qwen_2.5_vl_7b_fp8_scaled.safetensors",
    qwen_vae="qwen_image_vae.safetensors",
    qwen_lora="Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors",
    wan_high="Wan2.2-I2V-A14B-HighNoise-Q4_K_M.gguf",
    wan_low="Wan2.2-I2V-A14B-LowNoise-Q4_K_M.gguf",
    wan_clip="umt5-xxl-encoder-Q5_K_M.gguf",
    wan_vae="wan_2.1_vae.safetensors",
    wan_lora_high="wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors",
    wan_lora_low="wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors",
)

WAN_NEGATIVE = ("色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，"
                "低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，"
                "毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走")


class Graph:
    def __init__(self):
        self.nodes: dict[str, dict] = {}
        self.n = 0

    def add(self, class_type: str, **inputs) -> str:
        self.n += 1
        nid = str(self.n)
        self.nodes[nid] = {"class_type": class_type, "inputs": inputs}
        return nid

    @staticmethod
    def out(nid: str, idx: int = 0):
        return [nid, idx]


def _even16(v: int) -> int:
    return max(16, int(round(v / 16)) * 16)


def pov_frame(frame: Path, prompt: str, width: int, height: int, seed: int, prefix: str,
              steps: int = 4, cfg: float = 1.0, shift: float = 3.0) -> dict:
    """Qwen-Image-Edit-2509 edit of `frame`; output PNG saved under `prefix`."""
    g = Graph()
    model = g.out(g.add("UnetLoaderGGUF", unet_name=MODELS["qwen_unet"]))
    model = g.out(g.add("LoraLoaderModelOnly", model=model, lora_name=MODELS["qwen_lora"], strength_model=1.0))
    model = g.out(g.add("ModelSamplingAuraFlow", model=model, shift=shift))
    clip = g.out(g.add("CLIPLoader", clip_name=MODELS["qwen_clip"], type="qwen_image", device="default"))
    vae = g.out(g.add("VAELoader", vae_name=MODELS["qwen_vae"]))
    img = g.out(g.add("LoadImage", image=str(frame)))
    img = g.out(g.add("ImageScale", image=img, upscale_method="lanczos", width=_even16(width),
                      height=_even16(height), crop="center"))
    pos = g.out(g.add("TextEncodeQwenImageEditPlus", clip=clip, prompt=prompt, vae=vae, image1=img))
    neg = g.out(g.add("TextEncodeQwenImageEditPlus", clip=clip, prompt="", vae=vae, image1=img))
    latent = g.out(g.add("EmptySD3LatentImage", width=_even16(width), height=_even16(height), batch_size=1))
    latent = g.out(g.add("KSampler", model=model, positive=pos, negative=neg, latent_image=latent, seed=seed,
                         steps=steps, cfg=cfg, sampler_name="euler", scheduler="simple", denoise=1.0))
    out = g.out(g.add("VAEDecode", samples=latent, vae=vae))
    g.add("SaveImage", images=out, filename_prefix=prefix)
    return g.nodes


def outcome_clip(start: Path, prompt: str, width: int, height: int, length: int, seed: int, prefix: str,
                 fps: int = 16, steps: int = 4, cfg: float = 1.0, shift: float = 5.0, crf: int = 12) -> dict:
    """Wan2.2 I2V from `start`; two experts, 4 Lightning steps split 2 + 2; output mp4 under `prefix`."""
    g = Graph()
    high = g.out(g.add("UnetLoaderGGUF", unet_name=MODELS["wan_high"]))
    high = g.out(g.add("LoraLoaderModelOnly", model=high, lora_name=MODELS["wan_lora_high"], strength_model=1.0))
    high = g.out(g.add("ModelSamplingSD3", model=high, shift=shift))
    low = g.out(g.add("UnetLoaderGGUF", unet_name=MODELS["wan_low"]))
    low = g.out(g.add("LoraLoaderModelOnly", model=low, lora_name=MODELS["wan_lora_low"], strength_model=1.0))
    low = g.out(g.add("ModelSamplingSD3", model=low, shift=shift))
    clip = g.out(g.add("CLIPLoaderGGUF", clip_name=MODELS["wan_clip"], type="wan"))
    vae = g.out(g.add("VAELoader", vae_name=MODELS["wan_vae"]))
    pos = g.out(g.add("CLIPTextEncode", clip=clip, text=prompt))
    neg = g.out(g.add("CLIPTextEncode", clip=clip, text=WAN_NEGATIVE))
    img = g.out(g.add("LoadImage", image=str(start)))
    img = g.out(g.add("ImageScale", image=img, upscale_method="lanczos", width=_even16(width),
                      height=_even16(height), crop="center"))
    i2v = g.add("WanImageToVideo", positive=pos, negative=neg, vae=vae, width=_even16(width),
                height=_even16(height), length=length, batch_size=1, start_image=img)
    half = steps // 2
    lat = g.out(g.add("KSamplerAdvanced", model=high, add_noise="enable", noise_seed=seed, steps=steps, cfg=cfg,
                      sampler_name="euler", scheduler="simple", positive=g.out(i2v, 0), negative=g.out(i2v, 1),
                      latent_image=g.out(i2v, 2), start_at_step=0, end_at_step=half,
                      return_with_leftover_noise="enable"))
    lat = g.out(g.add("KSamplerAdvanced", model=low, add_noise="disable", noise_seed=seed, steps=steps, cfg=cfg,
                      sampler_name="euler", scheduler="simple", positive=g.out(i2v, 0), negative=g.out(i2v, 1),
                      latent_image=lat, start_at_step=half, end_at_step=10000,
                      return_with_leftover_noise="disable"))
    frames = g.out(g.add("VAEDecode", samples=lat, vae=vae))
    g.add("VHS_VideoCombine", images=frames, frame_rate=fps, loop_count=0, filename_prefix=prefix,
          format="video/h264-mp4", pix_fmt="yuv420p", crf=crf, save_metadata=False, trim_to_audio=False,
          pingpong=False, save_output=True)
    return g.nodes


def pov_prompt(character: str, scene: str, others: str) -> str:
    low = "cat" in character or "dog" in character or "animal" in character
    height = ("very low to the ground, the camera is only a few centimetres above the pavement" if low
              else "at that character's eye height")
    return (f"Change the camera to a first-person point-of-view shot seen through the eyes of {character}. "
            f"Remove {character} from the picture entirely: we ARE {character}, looking out of their eyes, "
            f"{height}, facing the direction they were facing. {character} must not be visible anywhere; "
            f"at most their own {'paws' if low else 'hands'} appear at the very bottom edge. "
            f"Keep the same place, lighting, moment, art style and colours, and keep everyone else "
            f"({others}) where they were, {scene}. Wide-angle, no text, no UI.")


def outcome_prompt(character: str, outcome: str, scene: str) -> str:
    return (f"First-person view from the eyes of {character}, {scene}. {outcome} The camera is their head: "
            f"it moves with their body, natural handheld motion, fast action, consistent lighting, "
            f"cinematic, no text.")
