# GaussianProcessTS — reference

## Paper

- **Title**: Gaussian Processes for Machine Learning
- **Venue**: MIT Press
- **Published**: 2006
- **Link**: https://gaussianprocess.org/gpml/chapters/

## Background

Gaussian process regression places a prior over functions and conditions kernel values on observations; the posterior mean at a query is `k_xz (K_zz + noise I)^-1 y_z`. The TSFLab baseline keeps only this posterior-mean form with learned inducing lag/forecast pairs.

## Citation

```bibtex
@book{DBLP:books/lib/RasmussenW06,
  author       = {Carl Edward Rasmussen and
                  Christopher K. I. Williams},
  title        = {Gaussian processes for machine learning},
  series       = {Adaptive computation and machine learning},
  publisher    = {{MIT} Press},
  year         = {2006},
  url          = {https://www.worldcat.org/oclc/61285753},
  isbn         = {026218253X},
  timestamp    = {Fri, 17 Jul 2020 16:12:42 +0200},
  biburl       = {https://dblp.org/rec/books/lib/RasmussenW06.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
