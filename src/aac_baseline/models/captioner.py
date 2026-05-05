import torch
import torch.nn as nn

from .common import EncoderOutput
from .decoder import BartCaptionDecoder, DecoderOutput
from .encoders import BeatsEncoderAdapter, ConvNextDummyEncoder
from .fusion import build_fusion_module
from .postencoder import ConformerPostEncoder


class DCASE24BaselineCaptioner(nn.Module):
    """
    Top-level model wiring.

    This file is intentionally boring:
    - call encoder 1
    - call encoder 2
    - fuse
    - refine with conformer
    - decode with BART

    """

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.config = config

        self.beats_encoder = BeatsEncoderAdapter(config["beats_encoder"])
        #this is where the rest of the encoders should be built, I have a dummy encoder for ConvNext but we should replace it with the 
        #correct one, and also add the AST
        self.convnext_encoder = ConvNextDummyEncoder(config["convnext_encoder"])
        #the following module is how we will fuse the encoder hidden states, 
        #right now it only supports fusing the beats and convnext dummy but we should also extend it to 
        #take as arguments the other encoders that we will implement
        self.fusion = build_fusion_module(
            fusion_config=config["fusion"],
            left_dim=config["beats_encoder"]["aggregation_hidden_size"],
            right_dim=config["convnext_encoder"]["hidden_size"],
        )
        self.postencoder = ConformerPostEncoder(config["conformer"])
        self.decoder = BartCaptionDecoder(config["bart"])

        self.fusion_stage = config["fusion"]["stage"]
        self.module_registry = {
            "beats_encoder": self.beats_encoder,
            "convnext_encoder": self.convnext_encoder,
            "fusion": self.fusion,
            "postencoder": self.postencoder,
            "decoder": self.decoder,
        }

    #the following section is for easily declaring which modules will be trainable
    #and which modules will be frozen, we can declare these things in the config.yml
    def set_module_trainable(self, module_name: str, trainable: bool) -> None:
        if module_name not in self.module_registry:
            raise KeyError(f"Unknown module name: {module_name}")

        for parameter in self.module_registry[module_name].parameters():
            parameter.requires_grad = trainable

    def encode_audio(
        self,
        waveforms: torch.Tensor,
        waveform_lengths: torch.Tensor,
        sample_rates: torch.Tensor,
    ) -> EncoderOutput:
        #here we are inferencing the encoders so we have the outputs of beats and the other encoders
        beats_output = self.beats_encoder(waveforms, waveform_lengths, sample_rates)
        convnext_output = self.convnext_encoder(waveforms, waveform_lengths, sample_rates)
        #and then we are passing them to the fusion mechanism
        if self.fusion_stage == "early":
            # Early fusion means we combine the encoder branches first and let
            # the conformer refine the already-fused representation.
            fused_output = self.fusion(beats_output, convnext_output)
            #after the fusion we are passign everything through the conformer
            return self.postencoder(fused_output)

        if self.fusion_stage == "late":
            # Late fusion means the BEATs branch is refined by the conformer
            # before we merge it with the second encoder branch.
            refined_beats = self.postencoder(beats_output)
            return self.fusion(refined_beats, convnext_output)

        raise ValueError(f"Unsupported fusion stage: {self.fusion_stage}")

    def forward(
        self,
        waveforms: torch.Tensor,
        waveform_lengths: torch.Tensor,
        sample_rates: torch.Tensor,
        labels: torch.Tensor,
    ) -> DecoderOutput:
        #this is the forward of the model
        encoder_output = self.encode_audio(waveforms, waveform_lengths, sample_rates)
        #the encode_audio function takes the input, passes it through all the encoders, fuses the encoders outputs
        #and then it passes it through the conformer, so the encoder_output is the output of the conformer 
        #then we pass it to the BART decoder
        return self.decoder(
            encoder_hidden_states=encoder_output.sequence,
            encoder_padding_mask=encoder_output.padding_mask,
            labels=labels,
        )

    @torch.no_grad()
    def score_captions(
        self,
        waveforms: torch.Tensor,
        waveform_lengths: torch.Tensor,
        sample_rates: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        encoder_output = self.encode_audio(waveforms, waveform_lengths, sample_rates)
        if labels.size(0) != encoder_output.sequence.size(0):
            if encoder_output.sequence.size(0) != 1:
                raise ValueError(
                    "score_captions can only expand a single encoded audio item "
                    f"to match labels, got audio batch={encoder_output.sequence.size(0)} "
                    f"and label batch={labels.size(0)}"
                )
            encoder_sequence = encoder_output.sequence.expand(labels.size(0), -1, -1)
            encoder_padding_mask = encoder_output.padding_mask.expand(labels.size(0), -1)
        else:
            encoder_sequence = encoder_output.sequence
            encoder_padding_mask = encoder_output.padding_mask

        return self.decoder.score_labels(
            encoder_hidden_states=encoder_sequence,
            encoder_padding_mask=encoder_padding_mask,
            labels=labels,
        )

    @torch.no_grad()
    def generate(
        self,
        waveforms: torch.Tensor,
        waveform_lengths: torch.Tensor,
        sample_rates: torch.Tensor,
        max_length: int = 32,
        min_length: int = 0,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        num_return_sequences: int = 1,
        no_repeat_ngram_size: int = 0,
    ) -> torch.Tensor:
        encoder_output = self.encode_audio(waveforms, waveform_lengths, sample_rates)
        return self.decoder.decode(
            encoder_hidden_states=encoder_output.sequence,
            encoder_padding_mask=encoder_output.padding_mask,
            max_length=max_length,
            min_length=min_length,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=num_return_sequences,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )
