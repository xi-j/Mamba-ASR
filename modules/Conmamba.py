"""ConMamba encoder and Mamba decoder implementation.

Authors
-------
* Xilin Jiang 2024
"""

import warnings
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import speechbrain as sb
from speechbrain.nnet.activations import Swish
from speechbrain.nnet.attention import (
    MultiheadAttention,
    PositionalwiseFeedForward,
    RelPosMHAXL,
)
from speechbrain.nnet.hypermixing import HyperMixing
from speechbrain.nnet.normalization import LayerNorm
from speechbrain.utils.dynamic_chunk_training import DynChunkTrainConfig

# Mamba
from mamba_ssm import Mamba
from modules.mamba.bimamba import Mamba as BiMamba 


class ConvolutionModule(nn.Module):
    """This is an implementation of convolution module in Conmamba.
    """

    def __init__(
        self,
        input_size,
        kernel_size=31,
        bias=True,
        activation=Swish,
        dropout=0.0,
        causal=False,
        dilation=1,
    ):
        super().__init__()

        self.kernel_size = kernel_size
        self.causal = causal
        self.dilation = dilation

        if self.causal:
            self.padding = (kernel_size - 1) * 2 ** (dilation - 1)
        else:
            self.padding = (kernel_size - 1) * 2 ** (dilation - 1) // 2

        self.layer_norm = nn.LayerNorm(input_size)
        self.bottleneck = nn.Sequential(
            # pointwise
            nn.Conv1d(
                input_size, 2 * input_size, kernel_size=1, stride=1, bias=bias
            ),
            nn.GLU(dim=1),
        )
        # depthwise
        self.conv = nn.Conv1d(
            input_size,
            input_size,
            kernel_size=kernel_size,
            stride=1,
            padding=self.padding,
            dilation=dilation,
            groups=input_size,
            bias=bias,
        )

        # BatchNorm in the original Conformer replaced with a LayerNorm due to
        # https://github.com/speechbrain/speechbrain/pull/1329
        # see discussion
        # https://github.com/speechbrain/speechbrain/pull/933#issuecomment-1033367884

        self.after_conv = nn.Sequential(
            nn.LayerNorm(input_size),
            activation(),
            # pointwise
            nn.Linear(input_size, input_size, bias=bias),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        dynchunktrain_config: Optional[DynChunkTrainConfig] = None,
    ):
        """Applies the convolution to an input tensor `x`.
        """

        if dynchunktrain_config is not None:
            # chances are chunking+causal is unintended; i don't know where it
            # may make sense, but if it does to you, feel free to implement it.
            assert (
                not self.causal
            ), "Chunked convolution not supported with causal padding"

            assert (
                self.dilation == 1
            ), "Current DynChunkTrain logic does not support dilation != 1"

            # in a causal convolution, which is not the case here, an output
            # frame would never be able to depend on a input frame from any
            # point in the future.

            # but with the dynamic chunk convolution, we instead use a "normal"
            # convolution but where, for any output frame, the future beyond the
            # "current" chunk gets masked.
            # see the paper linked in the documentation for details.

            chunk_size = dynchunktrain_config.chunk_size
            batch_size = x.shape[0]

            # determine the amount of padding we need to insert at the right of
            # the last chunk so that all chunks end up with the same size.
            if x.shape[1] % chunk_size != 0:
                final_right_padding = chunk_size - (x.shape[1] % chunk_size)
            else:
                final_right_padding = 0

            # -> [batch_size, t, in_channels]
            out = self.layer_norm(x)

            # -> [batch_size, in_channels, t] for the CNN
            out = out.transpose(1, 2)

            # -> [batch_size, in_channels, t] (pointwise)
            out = self.bottleneck(out)

            # -> [batch_size, in_channels, lc+t+final_right_padding]
            out = F.pad(out, (self.padding, final_right_padding), value=0)

            # now, make chunks with left context.
            # as a recap to what the above padding and this unfold do, consider
            # each a/b/c letter represents a frame as part of chunks a, b, c.
            # consider a chunk size of 4 and a kernel size of 5 (padding=2):
            #
            # input seq: 00aaaabbbbcc00
            # chunk #1:  00aaaa
            # chunk #2:      aabbbb
            # chunk #3:          bbcc00
            #
            # a few remarks here:
            # - the left padding gets inserted early so that the unfold logic
            #   works trivially
            # - the right 0-padding got inserted as the number of time steps
            #   could not be evenly split in `chunk_size` chunks

            # -> [batch_size, in_channels, num_chunks, lc+chunk_size]
            out = out.unfold(2, size=chunk_size + self.padding, step=chunk_size)

            # as we manually disable padding in the convolution below, we insert
            # right 0-padding to the chunks, e.g. reusing the above example:
            #
            # chunk #1:  00aaaa00
            # chunk #2:      aabbbb00
            # chunk #3:          bbcc0000

            # -> [batch_size, in_channels, num_chunks, lc+chunk_size+rpad]
            out = F.pad(out, (0, self.padding), value=0)

            # the transpose+flatten effectively flattens chunks into the batch
            # dimension to be processed into the time-wise convolution. the
            # chunks will later on be unflattened.

            # -> [batch_size, num_chunks, in_channels, lc+chunk_size+rpad]
            out = out.transpose(1, 2)

            # -> [batch_size * num_chunks, in_channels, lc+chunk_size+rpad]
            out = out.flatten(start_dim=0, end_dim=1)

            # TODO: experiment around reflect padding, which is difficult
            # because small chunks have too little time steps to reflect from

            # let's keep backwards compat by pointing at the weights from the
            # already declared Conv1d.
            #
            # still reusing the above example, the convolution will be applied,
            # with the padding truncated on both ends. the following example
            # shows the letter corresponding to the input frame on which the
            # convolution was centered.
            #
            # as you can see, the sum of lengths of all chunks is equal to our
            # input sequence length + `final_right_padding`.
            #
            # chunk #1:  aaaa
            # chunk #2:      bbbb
            # chunk #3:          cc00

            # -> [batch_size * num_chunks, out_channels, chunk_size]
            out = F.conv1d(
                out,
                weight=self.conv.weight,
                bias=self.conv.bias,
                stride=self.conv.stride,
                padding=0,
                dilation=self.conv.dilation,
                groups=self.conv.groups,
            )

            # -> [batch_size * num_chunks, chunk_size, out_channels]
            out = out.transpose(1, 2)

            out = self.after_conv(out)

            # -> [batch_size, num_chunks, chunk_size, out_channels]
            out = torch.unflatten(out, dim=0, sizes=(batch_size, -1))

            # -> [batch_size, t + final_right_padding, out_channels]
            out = torch.flatten(out, start_dim=1, end_dim=2)

            # -> [batch_size, t, out_channels]
            if final_right_padding > 0:
                out = out[:, :-final_right_padding, :]
        else:
            out = self.layer_norm(x)
            out = out.transpose(1, 2)
            out = self.bottleneck(out)
            out = self.conv(out)

            if self.causal:
                # chomp
                out = out[..., : -self.padding]

            out = out.transpose(1, 2)
            out = self.after_conv(out)

        if mask is not None:
            out.masked_fill_(mask, 0.0)

        return out


class ConmambaEncoderLayer(nn.Module):
    """This is an implementation of Conmamba encoder layer.
    """

    def __init__(
        self,
        d_model,
        d_ffn,
        kernel_size=31,
        activation=Swish,
        bias=True,
        dropout=0.0,
        causal=False,
        mamba_config=None
    ):
        super().__init__()
        assert mamba_config != None

        bidirectional = mamba_config.pop('bidirectional')
        if causal or (not bidirectional):
            self.mamba = Mamba(
                d_model=d_model,
                **mamba_config
            )
        else:
            self.mamba = BiMamba(
                d_model=d_model,
                bimamba_type='v2',
                **mamba_config
            )
        mamba_config['bidirectional'] = bidirectional

        self.convolution_module = ConvolutionModule(
            d_model, kernel_size, bias, activation, dropout, causal=causal
        )

        self.ffn_module1 = nn.Sequential(
            nn.LayerNorm(d_model),
            PositionalwiseFeedForward(
                d_ffn=d_ffn,
                input_size=d_model,
                dropout=dropout,
                activation=activation,
            ),
            nn.Dropout(dropout),
        )

        self.ffn_module2 = nn.Sequential(
            nn.LayerNorm(d_model),
            PositionalwiseFeedForward(
                d_ffn=d_ffn,
                input_size=d_model,
                dropout=dropout,
                activation=activation,
            ),
            nn.Dropout(dropout),
        )

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        pos_embs: torch.Tensor = None,
        dynchunktrain_config: Optional[DynChunkTrainConfig] = None,
    ):
        conv_mask: Optional[torch.Tensor] = None
        if src_key_padding_mask is not None:
            conv_mask = src_key_padding_mask.unsqueeze(-1)

        conv_mask = None

        # ffn module
        x = x + 0.5 * self.ffn_module1(x)
        # mamba module
        skip = x
        x = self.norm1(x)
        x = self.mamba(x)
        x = x + skip
        # convolution module
        x = x + self.convolution_module(
            x, conv_mask, dynchunktrain_config=dynchunktrain_config
        )
        # ffn module
        x = self.norm2(x + 0.5 * self.ffn_module2(x))
        return x


class ConmambaEncoder(nn.Module):
    """This class implements the Conmamba encoder.
    """

    def __init__(
        self,
        num_layers,
        d_model,
        d_ffn,
        kernel_size=31,
        activation=Swish,
        bias=True,
        dropout=0.0,
        causal=False,
        mamba_config=None
    ):
        super().__init__()
        print(f'dropout={str(dropout)} is not used in Mamba.')

        self.layers = torch.nn.ModuleList(
            [
                ConmambaEncoderLayer(
                    d_model=d_model,
                    d_ffn=d_ffn,
                    dropout=dropout,
                    activation=activation,
                    kernel_size=kernel_size,
                    bias=bias,
                    causal=causal,
                    mamba_config=mamba_config,
                )
                for i in range(num_layers)
            ]
        )
        self.norm = LayerNorm(d_model, eps=1e-6)

    def forward(
        self,
        src,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        pos_embs: Optional[torch.Tensor] = None,
        dynchunktrain_config: Optional[DynChunkTrainConfig] = None,
    ):
        """
        Arguments
        ----------
        src : torch.Tensor
            The sequence to the encoder layer.
        src_mask : torch.Tensor, optional
            The mask for the src sequence.
        src_key_padding_mask : torch.Tensor, optional
            The mask for the src keys per batch.
        pos_embs: torch.Tensor, torch.nn.Module,
            Module or tensor containing the input sequence positional embeddings
            If custom pos_embs are given it needs to have the shape (1, 2*S-1, E)
            where S is the sequence length, and E is the embedding dimension.
        dynchunktrain_config: Optional[DynChunkTrainConfig]
            Dynamic Chunk Training configuration object for streaming,
            specifically involved here to apply Dynamic Chunk Convolution to the
            convolution module.
        """

        output = src
        for enc_layer in self.layers:
            output = enc_layer(
                output,
                src_mask=src_mask,
                src_key_padding_mask=src_key_padding_mask,
                pos_embs=pos_embs,
                dynchunktrain_config=dynchunktrain_config,
            )
        output = self.norm(output)

        return output, None


class MambaDecoderLayer(nn.Module):
    """This class implements the Mamba decoder layer.
    """

    def __init__(
        self,
        d_model,
        d_ffn,
        activation=nn.ReLU,
        dropout=0.0,
        normalize_before=False,
        mamba_config=None
    ):
        super().__init__()

        assert mamba_config != None

        bidirectional = mamba_config.pop('bidirectional')

        self.self_mamba = Mamba(
            d_model=d_model,
            **mamba_config
        )

        self.cross_mamba = Mamba(
            d_model=d_model,
            **mamba_config
        )

        mamba_config['bidirectional'] = bidirectional

        self.pos_ffn = sb.nnet.attention.PositionalwiseFeedForward(
            d_ffn=d_ffn,
            input_size=d_model,
            dropout=dropout,
            activation=activation,
        )

        # normalization layers
        self.norm1 = sb.nnet.normalization.LayerNorm(d_model, eps=1e-6)
        self.norm2 = sb.nnet.normalization.LayerNorm(d_model, eps=1e-6)
        self.norm3 = sb.nnet.normalization.LayerNorm(d_model, eps=1e-6)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)
        self.dropout3 = torch.nn.Dropout(dropout)

        self.normalize_before = normalize_before

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        pos_embs_tgt=None,
        pos_embs_src=None,
    ):
        """
        Arguments
        ----------
        tgt: torch.Tensor
            The sequence to the decoder layer (required).
        memory: torch.Tensor
            The sequence from the last layer of the encoder (required).
        tgt_mask: torch.Tensor
            The mask for the tgt sequence (optional).
        memory_mask: torch.Tensor
            The mask for the memory sequence (optional).
        tgt_key_padding_mask: torch.Tensor
            The mask for the tgt keys per batch (optional).
        memory_key_padding_mask: torch.Tensor
            The mask for the memory keys per batch (optional).
        pos_embs_tgt: torch.Tensor
            The positional embeddings for the target (optional).
        pos_embs_src: torch.Tensor
            The positional embeddings for the source (optional).
        """
        if self.normalize_before:
            tgt1 = self.norm1(tgt)
        else:
            tgt1 = tgt

        # Mamba over the target sequence
        tgt2 = self.self_mamba(tgt1)

        # add & norm
        tgt = tgt + self.dropout1(tgt2)
        if not self.normalize_before:
            tgt = self.norm1(tgt)

        if self.normalize_before:
            tgt1 = self.norm2(tgt)
        else:
            tgt1 = tgt

        # Mamba over key=value + query
        # and only take the last len(query) tokens
        tgt2 = self.cross_mamba(torch.cat([memory, tgt1], dim=1))[:, -tgt1.shape[1]:]
        
        # add & norm
        tgt = tgt + self.dropout2(tgt2)
        if not self.normalize_before:
            tgt = self.norm2(tgt)

        if self.normalize_before:
            tgt1 = self.norm3(tgt)
        else:
            tgt1 = tgt

        tgt2 = self.pos_ffn(tgt1)

        # add & norm
        tgt = tgt + self.dropout3(tgt2)
        if not self.normalize_before:
            tgt = self.norm3(tgt)

        return tgt, None, None


class MambaDecoder(nn.Module):
    """This class implements the Mamba decoder.
    """

    def __init__(
        self,
        num_layers,
        d_model,
        d_ffn,
        activation=nn.ReLU,
        dropout=0.0,
        normalize_before=False,
        mamba_config=None
    ):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [
                MambaDecoderLayer(
                    d_model=d_model,
                    d_ffn=d_ffn,
                    activation=activation,
                    dropout=dropout,
                    normalize_before=normalize_before,
                    mamba_config=mamba_config
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = sb.nnet.normalization.LayerNorm(d_model, eps=1e-6)

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        pos_embs_tgt=None,
        pos_embs_src=None,
    ):
        """
        Arguments
        ----------
        tgt : torch.Tensor
            The sequence to the decoder layer (required).
        memory : torch.Tensor
            The sequence from the last layer of the encoder (required).
        tgt_mask : torch.Tensor
            The mask for the tgt sequence (optional).
        memory_mask : torch.Tensor
            The mask for the memory sequence (optional).
        tgt_key_padding_mask : torch.Tensor
            The mask for the tgt keys per batch (optional).
        memory_key_padding_mask : torch.Tensor
            The mask for the memory keys per batch (optional).
        pos_embs_tgt : torch.Tensor
            The positional embeddings for the target (optional).
        pos_embs_src : torch.Tensor
            The positional embeddings for the source (optional).
        """
        output = tgt
        for dec_layer in self.layers:
            output, _, _ = dec_layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos_embs_tgt=pos_embs_tgt,
                pos_embs_src=pos_embs_src,
            )
        output = self.norm(output)

        return output, [None], [None]
