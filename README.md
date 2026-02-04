# ComfyUI ACE15 Tiled Decode Audio

This node provides high-quality Audio Super Resolution (SR) for ComfyUI, powered by **FlashSR**. It features a **Tiled Processing** mechanism to handle long audio files without running out of VRAM.

## Features
- **Tiled Inference**: Automatically splits audio into chunks (245760 samples) to ensure stable U-Net alignment and low VRAM usage.
- **High Fidelity**: Supports upsampling to 44.1kHz and 48kHz.
- **Clean Logs**: Suppressed unnecessary library outputs for a cleaner ComfyUI console.

## Installation

1. Clone this repo to `custom_nodes`:
   ```bash
   git clone [https://github.com/klossm/ACE15_TiledDecodeAudio.git](https://github.com/klossm/ACE15_TiledDecodeAudio.git)