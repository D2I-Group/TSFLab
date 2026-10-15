# WaveNet — reference

## Paper

WaveNet: A Generative Model for Raw Audio (arXiv 1609.03499, 2016; ISCA SSW 2016).

A fully probabilistic, autoregressive deep network for raw audio waveforms, conditioning each sample's predictive distribution on all previous ones, yet trainable on tens of thousands of samples per second. It gave state-of-the-art text-to-speech in English and Mandarin, can switch between many speakers by conditioning on identity, generates realistic music fragments, and works as a discriminative model for phoneme recognition.

## Citation

```bibtex
@inproceedings{DBLP:conf/ssw/OordDZSVGKSK16,
  author    = {A{\"{a}}ron van den Oord and Sander Dieleman and Heiga Zen and
               Karen Simonyan and Oriol Vinyals and Alex Graves and
               Nal Kalchbrenner and Andrew W. Senior and Koray Kavukcuoglu},
  editor    = {Alan W. Black},
  title     = {WaveNet: {A} Generative Model for Raw Audio},
  booktitle = {The 9th {ISCA} Speech Synthesis Workshop, {SSW} 2016, Sunnyvale, CA,
               USA, September 13-15, 2016},
  pages     = {125},
  publisher = {{ISCA}},
  year      = {2016},
  url       = {https://www.isca-archive.org/ssw\_2016/vandenoord16\_ssw.html}
}
```
