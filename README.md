ACE15_TiledDecodeAudio (Updated)
A high-performance, memory-efficient 48kHz Tiled Audio Decoding node for ComfyUI. This node is specifically designed to handle long-duration audio latent samples by processing them in overlapping tiles, preventing VRAM out-of-memory (OOM) errors during VAE/Vocoder decoding.

🚀 Key Features & Implementation Logic
48kHz High-Fidelity Support: Optimized for high-resolution audio. The decoding pipeline is now strictly aligned with a 48000Hz sample rate for professional-grade output.

Intelligent Tiled Processing:

Auto-Alignment: Automatically aligns tile_size and overlap with internal CNN downsampling ratios (multiples of 64/32) to ensure seamless state transitions between blocks.

Boundary Smoothing: Implements a 50ms temporal fade-out at the end of the waveform to eliminate DC offset and clipping artifacts.

Advanced Audio Post-Processing:

Stereo Width Control: Adjusts the Mid/Side (M/S) balance to enhance or narrow the soundstage (Default: 1.15x for subtle enhancement).

HF Smoothing (Air-band): A specialized 15.5kHz treble biquad filter with a gentle Q-factor to polish the "air" frequencies and reduce digital harshness.

Adaptive Normalization: Uses a smart target-standard-deviation scaling (0.18) and final peak limiting (0.95) to ensure consistent loudness without clipping.

VRAM Efficiency: Features automatic model offloading to the intermediate device and active garbage collection (gc.collect) to keep ComfyUI running smoothly.

🛠️ Installation
Navigate to your ComfyUI custom_nodes folder:

Bash
cd ComfyUI/custom_nodes/
Clone the repository:

Bash
git clone https://github.com/klossm/ACE15_TiledDecodeAudio.git
Restart ComfyUI.