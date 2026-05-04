import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers.models.bart.modeling_bart import BartConfig
from transformers.models.wav2vec2_conformer.modeling_wav2vec2_conformer import Wav2Vec2ConformerConfig
from BEATs import BEATsConfig
from modeling_fused_conformer_bart import FusedEncodersConformerBartSeq2SeqForCaptioning

def test_fusion():
    # Setup configs
    try:
        from transformers import BartTokenizer
        tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
        vocab_size = len(tokenizer)
    except:
        tokenizer = None
        vocab_size = 1000

    config = BartConfig(
        vocab_size=vocab_size,
        d_model=768,
        encoder_layers=2,
        decoder_layers=2,
        encoder_attention_heads=4,
        decoder_attention_heads=4,
        encoder_ffn_dim=1024,
        decoder_ffn_dim=1024,
    )
    config.max_cross_position_embeddings = 1024
    config.lsm_weight = 0.1
    config.use_weighted_encoder_repr = True
    
    if tokenizer is not None:
        config.decoder_start_token_id = tokenizer.bos_token_id
        config.eos_token_id = tokenizer.eos_token_id
        config.pad_token_id = tokenizer.pad_token_id
    
    conformer_config = Wav2Vec2ConformerConfig(
        hidden_size=768,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=1024,
    )
    
    beats_config = BEATsConfig()
    beats_config.input_patch_size = 16
    beats_config.encoder_layers = 2
    beats_config.encoder_embed_dim = 768
    beats_config.encoder_ffn_embed_dim = 1024
    beats_config.encoder_attention_heads = 4
    
    # Initialize model
    print("Initializing Fused Model...")
    model = FusedEncodersConformerBartSeq2SeqForCaptioning(
        config=config,
        conformer_config=conformer_config,
        beats_config=beats_config
    )
    
    # Set model to eval
    model.eval()
    
    # Create dummy inputs
    bsize = 2
    # simulate a very short audio segment (e.g. 1 second at 16kHz = 16000 samples)
    enc_len = 16000 
    dec_len = 10
    
    encoder_input = torch.randn(bsize, enc_len)
    labels = torch.randint(0, config.vocab_size, (bsize, dec_len))
    attn_mask = torch.zeros(bsize, enc_len).bool()
    
    print("Running forward pass...")
    with torch.no_grad():
        out = model(
            encoder_input=encoder_input,
            labels=labels,
            attention_mask=attn_mask
        )
        
    print(f"Forward pass successful. Output loss: {out.loss}")
    print(f"Logits shape: {out.logits.shape}")
    print(f"Decoder hidden states returned: {out.decoder_hidden_states is not None}")

    print("\nTesting Text Generation Pipeline (from forward logits)...")
    try:
        if tokenizer is not None:
            # We take the argmax of the logits to simulate the predicted tokens
            predicted_ids = out.logits.argmax(dim=-1)
            
            print(f"Generated IDs shape: {predicted_ids.shape}")
            captions = tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)
            
            print("\nGenerated Mock Captions (will be gibberish since weights are random):")
            for i, cap in enumerate(captions):
                print(f"Caption {i+1}: {cap}")
        else:
            print("No tokenizer found. Skipping text decode.")
            
    except Exception as e:
        print(f"Generation test failed: {e}")

if __name__ == "__main__":
    test_fusion()
