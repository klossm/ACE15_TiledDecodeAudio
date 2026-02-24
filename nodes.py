import torch
import torchaudio
import sys
import os
import gc
import comfy.model_management

class ACE15_AudioDecoder:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "samples": ("LATENT",), 
                "vae": ("VAE",),
                "tile_size": ("INT", {"default": 1024, "min": 128, "max": 4096, "step": 128}), # Step matched to ratio logic
                "overlap": ("INT", {"default": 128, "min": 32, "max": 1024, "step": 32}),
                "denoise_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "stereo_width": ("FLOAT", {"default": 1.15, "min": 0.5, "max": 2.0, "step": 0.05}),
                "hf_smoothing": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "decode"
    CATEGORY = "ACE15/Audio"

    def decode(self, samples, vae, tile_size, overlap, denoise_strength, stereo_width, hf_smoothing):
        device = comfy.model_management.get_torch_device()
        offload_device = comfy.model_management.intermediate_device()
        
        if hasattr(vae, "first_stage_model"):
            vocoder_model = vae.first_stage_model
        else:
            vocoder_model = vae

        latent_samples = samples["samples"].to(device)
        latent_len = latent_samples.shape[-1]

        # 1920 logic alignment based on downsampling_ratios [2, 4, 4, 6, 10]
        # Although tile_size is in Latent space, alignment helps the internal CNN state
        tile_size = (tile_size // 64) * 64 
        overlap = (overlap // 32) * 32

        if tile_size > latent_len:
            tile_size = latent_len
        
        if overlap >= tile_size:
            overlap = tile_size // 4

        if denoise_strength > 0:
            latent_samples = torch.where(latent_samples.abs() < 1e-4, torch.zeros_like(latent_samples), latent_samples)
            latent_samples = latent_samples * (1.0 + (denoise_strength * 0.02))

        vocoder_model.to(device)
        try:
            with torch.no_grad():
                # Use 48000 as priority if available in config
                if hasattr(vae, "decode_tiled"):
                    audio = vae.decode_tiled(latent_samples, tile_y=tile_size, overlap=overlap)
                elif hasattr(vocoder_model, "decode_tiled"):
                    audio = vocoder_model.decode_tiled(latent_samples, tile_y=tile_size, overlap=overlap)
                else:
                    audio = vocoder_model.decode(latent_samples)
                
                audio = audio.movedim(-1, 1)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

        finally:
            vocoder_model.to(offload_device)

        audio = audio.to(torch.float32).cpu()
        
        # Enhanced Normalization logic
        eps = 1e-6
        std = torch.std(audio, dim=[1, 2], keepdim=True)
        if std.mean() > eps:
            # Smart scale: 48kHz content usually has higher energy density
            target_std = 0.18 
            scale = target_std / (std + eps)
            audio *= scale

        # Force 48000 from config
        sample_rate = 48000 

        if audio.shape[1] == 2 and stereo_width != 1.0:
            left, right = audio[:, 0:1, :], audio[:, 1:2, :]
            mid = (left + right) / 2.0
            side = (left - right) / 2.0
            side = side * stereo_width
            audio = torch.cat([mid + side, mid - side], dim=1)

        if hf_smoothing:
            # Adjusted for 48kHz: Center frequency moved to 15.5kHz for air-band
            audio = torchaudio.functional.treble_biquad(
                audio, 
                sample_rate=sample_rate, 
                gain=-2.5, 
                central_freq=15500.0, 
                Q=0.6
            )
            
            fade_samples = int(sample_rate * 0.05)
            if audio.shape[-1] > fade_samples:
                mask = torch.linspace(1.0, 0.0, fade_samples)
                audio[:, :, -fade_samples:] *= mask

        # Final peak normalization to prevent clipping
        max_val = torch.abs(audio).max()
        if max_val > 0.95:
            audio = audio * (0.95 / max_val)

        gc.collect()
        comfy.model_management.soft_empty_cache()

        return ({"waveform": audio, "sample_rate": sample_rate},)