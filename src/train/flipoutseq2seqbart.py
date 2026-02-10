from transformers import BartForConditionalGeneration

from src.train.flipoutseq2seqbase import FlipoutSeq2SeqBase


class FlipoutSeq2SeqBart(FlipoutSeq2SeqBase, BartForConditionalGeneration):
    """Implements BART with LinearFlipout layers. Not implemented currently."""
