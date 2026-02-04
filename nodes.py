import torch
import sys
import os

try:
    from comfy.ldm.ace.vae.music_vocoder import ADaMoSHiFiGANV1
except ImportError:
    import folder_paths
    base_path = os.path.join(os.path.dirname(folder_paths.__file__), "ldm", "ace", "vae")
    if base_path not in sys.path:
        sys.path.append(base_path)
    from music_vocoder import ADaMoSHiFiGANV1

class ACE15_AudioDecoder:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "samples": ("LATENT",), 
                "vae": ("VAE",),
                "tile_size": ("INT", {"default": 256, "min": 64, "max": 2048, "step": 64}),
                "overlap": ("INT", {"default": 32, "min": 8, "max": 256, "step": 8}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "decode"
    CATEGORY = "ACE15/Audio"

    def decode(self, samples, vae, tile_size, overlap):
        if hasattr(vae, "first_stage_model"):
            vocoder_model = vae.first_stage_model
        else:
            vocoder_model = vae
            
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        vocoder_model.to(device)
        vocoder_model.eval()

        # 1. Improved Sample Rate Detection
        sample_rate = getattr(vocoder_model, 'sampling_rate', 
                      getattr(vae, 'sampling_rate', 44100))

        target_dtype = next(vocoder_model.parameters()).dtype
        mel = samples["samples"].to(device=device, dtype=target_dtype)
        b, c, t = mel.shape
        
        # 2. Critical Fix: Accurate hop_length detection
        # We run two different sized mels to measure the output ratio
        with torch.no_grad():
            t1, t2 = 16, 32
            out1 = vocoder_model.decode(mel[:, :, :t1]).shape[-1]
            out2 = vocoder_model.decode(mel[:, :, :t2]).shape[-1]
            detected_hop = (out2 - out1) // (t2 - t1)
            out_channels = vocoder_model.decode(mel[:, :, :t1]).shape[1]
            
        hop_length = detected_hop
        
        # 3. Direct decode if small enough
        if t <= tile_size:
            with torch.no_grad():
                audio = vocoder_model.decode(mel)
            return ({"waveform": audio.cpu().float(), "sample_rate": sample_rate},)

        # 4. Tiled Decoding with Correct Shapes
        total_samples = t * hop_length
        output_audio = torch.zeros((b, out_channels, total_samples), device=device, dtype=torch.float32)
        weight_mask = torch.zeros((b, out_channels, total_samples), device=device, dtype=torch.float32)
        
        fade_len = overlap * hop_length
        window = torch.linspace(0, 1, fade_len, device=device, dtype=torch.float32).view(1, 1, -1)
        
        stride = tile_size - overlap
        
        for i in range(0, t, stride):
            start = i
            end = min(i + tile_size, t)
            if (end - start) < overlap and i > 0: # Skip if chunk is too small for overlap
                continue
                
            mel_chunk = mel[:, :, start:end]
            
            with torch.no_grad():
                chunk_audio = vocoder_model.decode(mel_chunk).to(torch.float32)
            
            chunk_l = chunk_audio.shape[-1]
            out_start = start * hop_length
            out_end = out_start + chunk_l
            
            # Constrain to buffer limits
            if out_end > total_samples:
                chunk_audio = chunk_audio[:, :, :total_samples - out_start]
                chunk_l = chunk_audio.shape[-1]
                out_end = total_samples

            chunk_weight = torch.ones((b, out_channels, chunk_l), device=device, dtype=torch.float32)
            
            # Linear crossfading
            if i > 0:
                chunk_weight[:, :, :fade_len] *= window
            if end < t:
                # Ensure we don't flip more than the chunk length
                current_fade_out = min(fade_len, chunk_l)
                chunk_weight[:, :, -current_fade_out:] *= torch.flip(window, [2])[:, :, -current_fade_out:]
            
            output_audio[:, :, out_start:out_end] += chunk_audio * chunk_weight
            weight_mask[:, :, out_start:out_end] += chunk_weight

        # Avoid division by zero and normalize
        final_audio = output_audio / (weight_mask + 1e-8)
        
        del output_audio, weight_mask
        torch.cuda.empty_cache()
        
        return ({"waveform": final_audio.cpu().float(), "sample_rate": sample_rate},)