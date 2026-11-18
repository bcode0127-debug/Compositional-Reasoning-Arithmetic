# Decisions

One entry per decision, newest last. An entry lands here after internal review and the author's approval.

## 2026-11-16: Repository layout and license

Context. Results lived in four places, the root mixed pre-v2 files with current ones, and the README's reproduction steps pointed at the pre-v2 pipeline. 175 lines in 34 tracked files reference `results_v2/`, including provenance recorded inside committed result files and the paper's Reproducibility paragraph.

Decision. Move every pre-v2 file under `legacy/`, keeping its original path after the prefix, and leave current paths unchanged. Rebuild the README around a table that links each reported number to its file and the command that produces it. Rename `results_v2/` to `results/` only after the paper's Reproducibility paragraph pins a commit. License everything under Apache-2.0: the code here, and the datasets and checkpoints on the Hugging Face Hub, where the dataset card already carries it.

Consequences. Paths written inside legacy files now point at their old locations, which `legacy/README.md` states. No committed result file is edited, so every path recorded in a current result stays true. `scripts/training/run_matrix.sh` stays unchanged as the record of how the 12 checkpoints were trained, so it still writes `matrix_summary.txt` at the repository root; the committed copy lives at `results_v2/matrix_summary.txt`.
