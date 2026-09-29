## LightGlue: Local Feature Matching at Light Speed
Slightly modified, self-contained, packaged version of https://github.com/cvg/LightGlue. This fork exists as a stable dependency for other scm tools.
If you are interested in LightGlue it might make sense to refer to the original repository.
Use at your own risk!

Weights that are not shipped with the package (e.g. LightGlue for SIFT) are downloaded on first use into the torch hub cache (`torch.hub.get_dir()`, configurable via `TORCH_HOME`).
