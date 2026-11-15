"""Word timestamps for greedy CTC that start where the word is emitted, not where the last one was.

NeMo's greedy CTC decoding (2.7.3, and unchanged on main as of 2026-10) records, per emitted
token, the gap since the *previous* emission (`token_lengths.append(pidx - last_length)`), and
`_compute_offsets` turns those gaps into the interval `[emission of token i-1, emission of token
i]`. Every token's span therefore ends where it fires and starts where the token before it fired,
so at the word level:

  * `start_offset` is the frame the PREVIOUS word's last token fired. After a pause the word
    start is dragged back across the whole silence: on NOTSOFAR (oracle masks, e16) word starts
    came out 0.22 s early at the median, 0.94 s on average, and 16% of them more than 0.5 s early
    against the forced alignments.
  * `end_offset` is the last token's emission frame, with no frame of its own: 0.22 s early at the
    median.

`end_offset` of each *token* is still the true emission frame, so the words are rebuilt from those:
a word spans from its first token's emission to one frame past its last token's. Against the same
alignments that gives starts at +0.00 s median (p10/p90 -0.09/+0.09 s) and ends at -0.14 s; the
remaining end bias is CTC's peakiness (a word occupies the frames it fires on, not its full sound)
and is left alone.
"""

from typing import Iterable

from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis

__all__ = ['emission_word_offsets']


def emission_word_offsets(hypotheses: Iterable[Hypothesis]):
    """Rewrite each hypothesis's `timestamp['word']` offsets in place to the emission convention.

    Hypotheses without char/word timestamps (plain text, or `compute_timestamps` off) pass through
    untouched. Returns the same iterable for chaining.
    """
    for hypothesis in hypotheses:
        timestamp = getattr(hypothesis, 'timestamp', None)
        if not isinstance(timestamp, dict) or not timestamp.get('word') or not timestamp.get('char'):
            continue
        # A token's span is (previous emission, own emission], so the tokens of a word are those
        # whose emission falls in (word start, word end].
        emissions = sorted(int(c['end_offset']) for c in timestamp['char'])
        k = 0
        for word in timestamp['word']:
            start, end = int(word['start_offset']), int(word['end_offset'])
            while k < len(emissions) and emissions[k] <= start:
                k += 1
            first = emissions[k] if k < len(emissions) and emissions[k] <= end else end
            word['start_offset'] = first
            word['end_offset'] = end + 1
    return hypotheses
