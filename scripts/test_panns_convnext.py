import torch
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from aac_baseline.models.encoders.convnext_panns import ConvNextEncoderAdapter

def test_loading():
    print("--- Testing ConvNextEncoderAdapter Loading ---")
    
    config = {
        "pretrained_checkpoint_path": "/Users/jonathandavid/NYU/Spring 2026/Machine Listening/FinalProject/ConvNextAudioCheckpoints/convnext_tiny_471mAP.pth",
        "target_sample_rate": 32000,
        "hidden_size": 768,
        "hop_size": 320,
        "after_stem_dim": [252, 56],
    }
    
    try:
        encoder = ConvNextEncoderAdapter(config)
        print("✅ Encoder initialized and checkpoint loaded successfully!\n")
        
        # Test forward pass with dummy audio (1 second at 32kHz)
        batch_size = 2
        duration = 1.0
        sample_rate = 32000
        waveforms = torch.randn(batch_size, int(duration * sample_rate))
        lengths = torch.tensor([int(duration * sample_rate)] * batch_size)
        rates = torch.tensor([sample_rate] * batch_size)
        
        print(f"Input Audio Shape: {waveforms.shape} (2 files, 32,000 samples)")
        
        output = encoder(waveforms, lengths, rates)
        
        print(f"✅ Forward pass successful!\n")
        print(f"Output Sequence Shape: {output.sequence.shape}")
        print(f"Padding Mask Shape:    {output.padding_mask.shape}")
        
        print(f"\nNotice how the Time dimension went from 32,000 down to {output.sequence.shape[1]}!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_loading()
