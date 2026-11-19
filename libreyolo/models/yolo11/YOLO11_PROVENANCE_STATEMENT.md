# Provenance statement — experimental YOLO11n attention and detection towers

## Contribution and authorship

This statement covers:

| Component | Files | Introduced in | Later revisions |
|---|---|---|---|
| C2PSA attention interface | `libreyolo/models/yolo11/attention.py`, `tests/unit/test_yolo11_attention.py` | `0912a67` | `a8b1371`, `19bb4a7`, `ed595b6`, `aa6f764`, `1894c4e` |
| Box and classification towers | `libreyolo/models/yolo11/detect.py`, `tests/unit/test_yolo11_detect.py` | `0d78bfc` | `a8b1371`, `19bb4a7`, `ed595b6`, `aa6f764`, `1894c4e`, `3d74ab4`, `a109678`, `238545e`, `4c4c756`, `536d5e6`, `c414e2d` |
| Experimental nano preset 0.1.0 | `libreyolo/models/yolo11/presets.py`, `tests/unit/test_yolo11_presets.py` | `1894c4e` | — |

I directed the coding agent to implement the attention module, detection towers,
and experimental preset. The agent wrote the code and tests. I reviewed its
summaries and reported test results.

The handoff was supplied after the initial drafts and before the rewrite. It is
saved on 2026-11-18 as
`AppData/Local/hermes/composer-pastes/pasted_content_2026-11-18_05-55-43-537_3b6312.txt`,
for branch `ps2chiper/libreyolo:yolo11-native-port`. The handoff identified the
baseline paper sections and native LibreYOLO files, and instructed contributors
not to use the branch's `PORT.md` or the earlier audit conversation because they
contain observations from provenance-disputed implementations.

## Attention

The port notes cite Meng et al., Figure 2a and Section 2, for the C2PSA outer
structure — an initial convolution, cascaded PSA blocks, a bypass path, a
concatenation and a final convolution — and for residual connections in the PSA
blocks. They cite Saeheaw, Algorithm A1, line 20, for the stage width of 256 and
the outer expansion of 0.5. The implementation uses LibreYOLO's native `Conv`
block with BatchNorm and SiLU.

`C2PSA` takes `blocks` and `split_ratio` separately. `PSATopology` requires the
remaining values the sources do not establish. Preset 0.1.0 fixes them as
follows:

| Choice | Value | Basis |
|---|---|---|
| Stage width, outer expansion | 256, 0.5 | Saeheaw, Algorithm A1 |
| Block repeat count | 1 | Experimental; A1 illustrates `n=2` |
| Split ratio | 0.5 | Experimental |
| Attention heads | 3 | Experimental |
| Per-head query/key/value widths | 16 / 16 / 32 | Experimental |
| QKV packing | `head_major` | Experimental; the packing order is not established by the sources |
| Position term | 3×3 depthwise convolution, added to the value map before attention mixes spatial sites | The depthwise operation is fixed in code; its placement is an experimental LibreYOLO choice |
| Feed-forward path | 1×1 convolution, expansion 2, no output activation | Experimental |
| Residuals | Both sublayers | Experimental |

The coding agent proposed the attention and detection tower values. I approved
the experimental preset as a whole. I did not select or approve individual
values.

The only source-backed attention
values are the 256 stage width, the 0.5 outer expansion, and the resulting
256-channel input/output shape. The repeat count, split, head layout, packing,
position placement, feed-forward form and residual placement are LibreYOLO
design decisions. Baseline QKV packing and positional placement remain unverified.

## Detection towers

The port notes cite Feng et al., Figure 1, for separate box and classification
branches with two convolution stages each. They cite Wang et al., Section 2.2.1,
for two depthwise convolutions in the classification branch. They cite Saeheaw,
Algorithm A1, for the three input widths of 64, 128 and 256 channels. The output
follows LibreYOLO's native decoder contract: `(B, 64 + C, H, W)`, with four
consecutive 16-bin groups in left, top, right, bottom order and one raw class
logit per class.

Preset 0.1.0 fixes the open tower values as follows:

| Choice | Value | Basis |
|---|---|---|
| Box tower width | 48 | Experimental |
| Classification tower width | 40, fixed and not scaled with class count | Experimental |
| Kernels | 3×3 in both towers | Experimental |
| Box grouping | 1 group (plain convolution) | Experimental |
| Classification stage order | depthwise then pointwise | Wang establishes two DWConv layers; the full order is experimental |
| Pointwise and output projections | 1×1 | Experimental |
| Output bias | Enabled | Experimental |

The widths, kernels, groupings and projections in this table were among the
experimental choices proposed by the coding agent. The sources establish
the two-branch, two-stage arrangement, the presence of depthwise convolutions in
the classification branch, and the 64/128/256 input widths; they do not establish
how to reproduce the baseline values.

## Source exposure and reuse

### Papers and publisher copies

`PORT.md` cites the following passages and figures from the archived publisher
HTML/PDF copies. The branch record establishes the citations; it does not
establish who personally read each source. The primary sources for these modules
were:

- Saeheaw (2026), *Comparative Evaluation of YOLOv11n Neck-Level Modifications
  for Precast Component and PPE Object Detection in Construction Environments*,
  Appendix B, Algorithm A1. CC BY 4.0.
- Meng et al. (2025), *Enhanced YOLOv11 Framework for Accurate Multi-Fault
  Detection in UAV Photovoltaic Inspection*, Figure 2a and Section 2. CC BY 4.0.
- Feng et al. (2026), *HIFE-YOLO: Hierarchical Information Flow Enhancement for
  Lightweight Fire and Smoke Detection*, baseline Figure 1. CC BY 4.0.
- Wang et al. (2025), *Tomato Yield Estimation Using an Improved Lightweight
  YOLO11n Network and an Optimized Region Tracking-Counting Method*, Section
  2.2.1. CC BY 4.0.

`PORT.md` also cites Xu et al. (2025), Table 1, and Wei et al. (2026), Table 2,
for the published scale table, but they do not establish the preset's open
values. The wording in the port notes paraphrases these CC BY 4.0 descriptions;
no text or figure was reproduced as code.

I relied on the coding agent's summaries for all six papers. I did not
personally read any of the papers or view their figures or tables.

### Native LibreYOLO material

The following LibreYOLO components and project files are relevant to the implementation and tests.

- `libreyolo/models/yolo9/nn.py`, especially `Conv` (BatchNorm + SiLU), which the
  attention block and towers use.
- `libreyolo/models/yolo11/decoder.py`, the native `YOLO11nDecoder` defining the
  raw-map contract.
- `libreyolo/models/yolo11/transforms.py` and the existing decoder and tower
  tests for the LibreYOLO conventions used in the new tests.
- `AGENTS.md`, `pyproject.toml`, and the repository's test conventions.

The audit compared attention.py, detect.py and presets.py against the inspected
Ultralytics snapshot. It found no verbatim match of four consecutive qualifying
lines totaling at least 110 characters after excluding blank lines, full-line
comments and imports. The earlier checkpoint inspection and the overlap
between drafts are documented below. How much that exposure influenced the final
implementation remains uncertain.

The native `Conv` ultimately derives from LibreYOLO's YOLO9 work;
that upstream and its MIT licence are identified in the repository's
`THIRD_PARTY_NOTICES.txt` and `libreyolo/models/yolo11/NOTICE`.

I did not personally inspect any of the native LibreYOLO files or their diffs.
I relied on the coding agent's summaries and reported test results.

### Earlier implementation exposure

Before the handoff, the implementing agent reviewed the port audit and earlier
conversation and inspected module names, attribute names, and tensor shapes from
an official YOLO11n checkpoint. That inspection informed the first attention and
detection drafts (`cdc5b0b`, `cbc8b2d`, `e9a1e67`). Those commits are outside the
port branch history. The later implementations (`0912a67`, `0d78bfc`)
parameterized details the handoff left open. The two attention drafts share some
names and expressions. I cannot determine how much the earlier exposure
influenced the later implementation. The concrete preset was introduced
separately in `1894c4e`. The port notes identify its experimental choices and
the dimensions supported by the cited papers.

A diff of the discarded attention file against `0912a67` shows observable
continuity: both use the `PSAAttention` and `PSABlock` class names and the
`qkv`, `proj` and `pe` attribute names; the QKV construction and two
residual-update lines match verbatim. That establishes overlap between the two
drafts. It does not establish copying from Ultralytics.

The two detection drafts also share structure. Both build separate box and
classification towers for each input level. Both classification towers use two
depthwise-then-pointwise stages followed by an output projection. The later draft
replaces the earlier width formulas with explicit topology parameters. The cited
papers support separate branches and two classification depthwise convolutions.
The extent to which the earlier draft influenced the retained stage order is
uncertain.

The cited papers independently support some of the same dimensions, including
the 256-channel stage width. Git cannot establish which information a
contributor relied on when choosing any particular value.

I read the audit conversation and the coding agent's summaries. I passed the
handoff to the agent. I did not personally inspect the checkpoint output or
external source code, supply code snippets, or suggest tower values.

## Right to contribute

I confirm I have the right to submit the AI-assisted contributions described in the statement to LibreYOLO under its MIT license.
