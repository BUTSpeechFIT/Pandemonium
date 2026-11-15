"""CTC output projection built from a pretrained transducer joint.

NeMo's joint is `joint_net(enc(f) + pred(g))`; holding `g` at a constant collapses the `U` axis
and leaves a frame-wise classifier that starts from Parakeet's trained output layer instead of
a random one. The prediction network is dropped.

Two things must not change: the module names and `joint_net` layer order mirror `RNNTJoint`, so
a checkpoint loads after a prefix rename (dropping the dropout would move the final Linear from
index 2 to 1 and silently break it); and a TDT joint emits `V + 1 + durations` logits, so the
durations are sliced off before the log-softmax, leaving the transducer blank at index `V` --
where NeMo's CTC decoding expects it.
"""

import torch
import torch.nn as nn

from nemo.core.classes import NeuralModule
from nemo.core.neural_types import AcousticEncodedRepresentation, LogprobsType, NeuralType

__all__ = ['JointCTCDecoder']


class JointCTCDecoder(NeuralModule):
    """Frame-wise CTC projection with the parameter layout of `RNNTJoint`.

    Args:
        feat_in: Encoder output width (`enc_hidden`).
        num_classes: Vocabulary size *excluding* the blank, as in the pretrained joint.
        pred_hidden: Width of the prediction network the pretrained joint was built against.
            Only its projection is kept; the network itself is gone.
        joint_hidden: Width of the joint's hidden step.
        activation: Joint activation, one of relu/sigmoid/tanh.
        dropout: Joint dropout. Must match the pretrained joint; see the module docstring.
        num_extra_outputs: TDT duration outputs to allocate and then ignore. Needed so the final
            layer has the same width as the checkpoint's.
        vocabulary: Token list, kept for NeMo's decoding to read off.
    """

    @property
    def input_types(self):
        return {"encoder_output": NeuralType(('B', 'D', 'T'), AcousticEncodedRepresentation())}

    @property
    def output_types(self):
        return {"logprobs": NeuralType(('B', 'T', 'D'), LogprobsType())}

    def __init__(
        self,
        feat_in: int,
        num_classes: int,
        pred_hidden: int = 640,
        joint_hidden: int = 640,
        activation: str = 'relu',
        dropout: float = 0.2,
        num_extra_outputs: int = 5,
        vocabulary=None,
    ):
        super().__init__()

        if vocabulary is not None and num_classes != len(vocabulary):
            raise ValueError(
                f"num_classes ({num_classes}) does not match the vocabulary ({len(vocabulary)})."
            )
        self.__vocabulary = vocabulary
        self._num_classes_with_blank = num_classes + 1

        self.pred = nn.Linear(pred_hidden, joint_hidden)
        self.enc = nn.Linear(feat_in, joint_hidden)

        if activation not in ('relu', 'sigmoid', 'tanh'):
            raise ValueError(f"Unsupported joint activation {activation!r}; use relu/sigmoid/tanh.")
        activation_module = {'relu': nn.ReLU(inplace=True), 'sigmoid': nn.Sigmoid(), 'tanh': nn.Tanh()}[activation]

        layers = [activation_module]
        if dropout:
            layers.append(nn.Dropout(p=dropout))
        layers.append(nn.Linear(joint_hidden, self._num_classes_with_blank + num_extra_outputs))
        self.joint_net = nn.Sequential(*layers)

        # Stand-in for the prediction network's output. Passed through `self.pred` rather than
        # reading `pred.bias`, so `pred.weight` stays in the graph -- a parameter that never
        # participates makes DDP raise. Its gradient is zero, so the projection is untouched.
        self.register_buffer('sos_embedding', torch.zeros(1, 1, pred_hidden), persistent=False)

    @property
    def num_classes_with_blank(self) -> int:
        return self._num_classes_with_blank

    @property
    def vocabulary(self):
        return self.__vocabulary

    def forward(self, encoder_output: torch.Tensor) -> torch.Tensor:
        """(B, D, T) encoder output -> (B, T, V + 1) log-probabilities."""
        f = self.enc(encoder_output.transpose(1, 2))  # (B, T, joint_hidden)
        g = self.pred(self.sos_embedding.to(f.dtype))  # (1, 1, joint_hidden)

        logits = self.joint_net(f + g)  # (B, T, V + 1 + durations)
        logits = logits[..., : self._num_classes_with_blank]  # drop the TDT duration outputs

        return torch.log_softmax(logits, dim=-1)
