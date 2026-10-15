# Third-Party Notices

TSFLab maintains every registered model as local repository code. Papers and
official repositories are cited to document provenance and implementation
decisions; those references do not make external model source part of this
distribution.

Each model card at `src/tsflab/models/<model>/README.md` records the paper and, when an
official implementation exists, its repository URL, pinned revision, and license
label. Consult that repository at the recorded revision for its complete license
and notices. A model card with `codebase: null` has no identified official
codebase.

Some official repositories publish no license. For those models the pinned
repository was consulted only as a reference for paper details; the TSFLab
implementation is an independent rewrite from the paper under the project
license, contains no code copied from that repository, and its card records the
missing license rather than assuming one.

Runtime Python dependencies are declared in `pyproject.toml` and retain their own
licenses and notices. TSFLab does not vendor their source. Built wheels contain
the TSFLab runtime packages and curated Agent assets, not external model
repositories, test suites, local datasets, or downloaded model artifacts.
