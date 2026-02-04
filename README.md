ACE15_TiledDecodeAudio
A high-performance, memory-efficient Tiled Audio Decoding node for ComfyUI. This node is specifically designed to handle long-duration audio latent samples that would otherwise exceed VRAM limits during VAE/Vocoder decoding.

🚀 Key Features & Implementation Logic
1. Dynamic Hop-length Detection
Unlike standard decoders that rely on hardcoded values, this node implements an Auto-Precision Calibration logic.

The Method: It performs two micro-decodes of different temporal lengths to mathematically calculate the exact hop_length of the loaded model.

The Result: Seamless compatibility with various HiFi-GAN and Audio VAE architectures without manual user configuration.

2. Tiled Processing with Linear Crossfading
To process extremely long audio files:

Tiling: The Latent space is processed in segments defined by tile_size.

Seamless Blending: Uses a weight_mask and torch.linspace window to perform linear crossfading between overlapping chunks. This eliminates "clicking" or "popping" artifacts at the segment boundaries.

Normalization: Every sample is normalized by the accumulated weight mask to ensure consistent volume across the entire waveform.

3. Smart Fallback & Robustness
Automatic Pathing: Includes a robust import system that searches for music_vocoder dependencies within the ComfyUI directory structure.

VRAM Optimization: Forces evaluation mode and automatically cleans the CUDA cache after processing to keep your workflow smooth.


🛠️ Installation
Navigate to your ComfyUI custom_nodes folder:

Bash
cd ComfyUI/custom_nodes/
Clone the repository:

Bash
git clone https://github.com/klossm/ACE15_TiledDecodeAudio.git
Restart ComfyUI.