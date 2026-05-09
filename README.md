# fast-ctc-decode

![test-fast-ctc-decode](https://github.com/nanoporetech/fast-ctc-decode/workflows/test-fast-ctc-decode/badge.svg) [![PyPI version](https://badge.fury.io/py/fast-ctc-decode.svg)](https://badge.fury.io/py/fast-ctc-decode)

- Forked from [fast-ctc-decode](https://github.com/nanoporetech/fast-ctc-decode) (v0.3.6) by ONT
- Based on the work `SynDe: Syndrome--guided Decoding of Raw Nanopore Reads` [Arxiv: ]
- The `my-extension` branch of this repository includes the CTC-based implementations of our novel algorithms PrimerSeeker and Synde.
    - PrimerSeeker: a dedicated algorithm that locates the start of a primer in the raw read
    - Synde: a solution for basecaller-decoder integration that performs convolutional decoding by performing a constrained beam search -  one that exploits the syndrome trellis representation of the concerned convolutional code. Its main advantage is that its complexity is independent of the memory of the convolutional code.
- Thanks to [Roman Sokolovskii](https://github.com/rsokolovskii) for contributing to this project!!

## Download and installation

```bash
git clone --recursive -b my-extension https://github.com/anisha-ban/fast-ctc-decode-synde-primerseeker
cd fast-ctc-decode-synde-primerseeker/
pip install "maturin>=0.14,<0.15"
python -m maturin build --release --features python # this should create a folder `target` named target with the required wheel file
pip install target/wheels/*.whl --force-reinstall
cd ..
```

## Usage

- The original functionality (beam_search, viterbi_search, ...) are preserved and the information on the use of these functions can be found in `old-README.md`.

```python
>>> from fast_ctc_decode import primer_beam_search_opt as primer_beam_search
>>> from fast_ctc_decode import convolutional_beam_search_log, marker_beam_search_log_track
>>> import numpy as np
>>> import json
>>> alphabet = "NACGT"
>>> prob_matrix=np.matrix([[0.4, 0.6, 0.0,  0,   0],
                           [  0, 0.2, 0.8,  0,   0],
                           [  0,   0, 1.0,  0,   0],
                           [  0,   0,   1,  0,   0],
                           [  0,   0,   0,  1,   0],
                           [  0, 0.5,   0,  0, 0.5],
                           [  0,   0,   1,  0,   0],
                           [  0,   0,   0,  1,   0],
                           [  0,   0,   0,  0,   1]],dtype='float32')

>>> primer = "ACGT"
>>> beam = 8
>>> fraction_to_examine = 1.0
>>> beam_cut_threshold = 0.0
>>> concentration_threshold = 1.0
>>> max_sample_depth = 6
>>> shift = 2 # just for this example. normally we set this to 100
>>> subsample = 1
>>> primer_scores, total_comp = primer_beam_search(prob_matrix, beam, fraction_to_examine, beam_cut_threshold, concentration_threshold, primer, max_sample_depth, shift, subsample)
>>> primer_location = np.argmax(np.array(primer_scores))
>>> primer_location
5

>>> with open("convolutional_codes_trellises/cc_2_1_3.json") as f:
>>>    conv_config = json.load(f)
>>> prob_matrix=np.matrix([ [0.0, 1.0, 0.0, 0.0, 0.0],  # A - 0
                            [0.0, 0.0, 0.0, 0.0, 1.0],  # T - 1
                            [0.0, 0.0, 0.0, 0.0, 1.0],  # T - 2  AT (2)
                            [0.0, 1.0, 0.0, 0.0, 0.0],  # A - 3  ATA (3)
                            [1.0, 0.0, 0.0, 0.0, 0.0],  # N - 4  ATA (3)
                            [0.0, 1.0, 0.0, 0.0, 0.0],  # A - 5  ATAA (4)
                            [0.0, 0.0, 1.0, 0.0, 0.0],  # C - 6  ATAAC (5)
                            [0.0, 0.0, 1.0, 0.0, 0.0],  # C - 7  ATAAC (5)
                            [1.0, 0.0, 0.0, 0.0, 0.0],  # N - 8  ATAAC (5)
                            [0.0, 0.0, 1.0, 0.0, 0.0],  # C - 9  ATAACC (6)
                            [0.0, 0.0, 1.0, 0.0, 0.0],  # C - 10 ATAACC (6)
                            [0.0, 0.0, 0.0, 0.0, 1.0],  # T - 11 ATAACCT
                            [0.0, 0.0, 1.0, 0.0, 0.0],  # C
                            [0.0, 0.0, 0.0, 1.0, 0.0],  # G
                        ], dtype='float32')

>>> forward_primer_str = "AT"
>>> reverse_primer_str = "CG"
>>> offset_sequence_str = "CTAAA"
>>> seq, starts, score, total_computations = convolutional_beam_search_log(
                                                    prob_matrix,
                                                    alphabet,
                                                    beam_size=5,
                                                    beam_cut_threshold=0.0,
                                                    collapse_repeats=True,
                                                    forward_primer_str=forward_primer_str,
                                                    reverse_primer_str=reverse_primer_str,
                                                    offset_sequence_str=offset_sequence_str,
                                                    conv_config=conv_config,
                                                )
>>> seq # works since [3,1,1,1,3] or 'TCCCT' is a codeword of the [2,1,3] conv code. Payload 'AACCT' is codeword+offset (mod 4)
'ATAACCTCG'

>>> prob_matrix=np.matrix([ [0.0, 1.0, 0.0, 0.0, 0.0],  # A - 0
                            [0.0, 0.0, 0.0, 0.0, 1.0],  # T - 1
                            [0.0, 0.0, 0.0, 0.0, 1.0],  # T - 2  AT (2)
                            [0.0, 1.0, 0.0, 0.0, 0.0],  # A - 3  ATA (3)
                            [1.0, 0.0, 0.0, 0.0, 0.0],  # N - 4  ATA (3)
                            [0.0, 0.0, 1.0, 0.0, 0.0],  # C - 5  ATAC (4)
                            [0.0, 0.0, 1.0, 0.0, 0.0],  # C - 6  ATAC (5)
                            [0.0, 0.0, 1.0, 0.0, 0.0],  # C - 7  ATAC (5)
                            [1.0, 0.0, 0.0, 0.0, 0.0],  # N - 8  ATAC (5)
                            [0.0, 0.0, 1.0, 0.0, 0.0],  # C - 9  ATACC (6)
                            [0.0, 0.0, 1.0, 0.0, 0.0],  # C - 10 ATACC (6)
                            [0.0, 0.0, 0.0, 0.0, 1.0],  # T - 11 ATACCT
                            [0.0, 1.0, 0.0, 0.0, 0.0],  # A - 12 ATACCTA
                            [0.0, 0.0, 1.0, 0.0, 0.0],  # C
                            [0.0, 0.0, 0.0, 1.0, 0.0],  # G
                        ], dtype='float32')
>>> forward_primer_str = "AT"
>>> reverse_primer_str = "CG"
>>> offset_sequence_str = "AAAAA"
>>> marker_interval = 2
>>> marker_sequence_str = "C"
>>> seq, score, base_probabilities = marker_beam_search_log_track(
                                                    prob_matrix,
                                                    alphabet,
                                                    beam_size=5,
                                                    beam_cut_threshold=0.0,
                                                    collapse_repeats=True,
                                                    forward_primer_str=forward_primer_str,
                                                    reverse_primer_str=reverse_primer_str,
                                                    offset_sequence_str=offset_sequence_str,
                                                    marker_interval=marker_interval,
                                                    marker_sequence_str=marker_sequence_str,
                                                ) # seq should be 'ATACCTACG'
```
