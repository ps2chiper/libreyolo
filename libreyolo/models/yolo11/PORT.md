# YOLO11 detection port

This branch implements the image and detection boundary using LibreYOLO's native
preprocessing, letterbox geometry and class-aware postprocessing. A caller supplies
RGB images and a target canvas, then supplies decoded predictions in the native
pixel-space xyxy format. Centered padding is a choice for this port. Synthetic
tests exercise the boundary independently of any model or checkpoint.

## Provenance decision needed

Earlier Glassbox work involved comparison with GPL/AGPL implementations. The
affected areas include the model graph and scaling, C3k2 and C2PSA/attention
stack, detection-head layout and decoding, and several training routines.
The maintainers should choose an independently documented permissive source or
omit the affected behavior. The family architecture and head need an independent
implementer under LibreYOLO's provenance policy. For training, prefer native
assignment, EMA, mosaic, batching and evaluation. Exact legacy behavior would
require a separate provenance decision.

The next integration milestone is a scratch-initialized detector that produces
the adapter's decoded input. Factory registration, training, validation and
export support should follow tests of that detector. This branch contains no
family weights or download entry.

## Published scale presets

The values below appear in Table 1 of both Xu et al. and Wei et al. The table
is reformatted in ascending size order. Both papers publish it under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). These presets
leave the order of channel capping and rounding, repeat rounding, and
size-dependent block choices to be established separately.

| YOLO11 size | Depth multiplier | Width multiplier | Maximum channels |
|---|---:|---:|---:|
| Nano (n) | 0.50 | 0.25 | 1024 |
| Small (s) | 0.50 | 0.50 | 1024 |
| Medium (m) | 0.50 | 1.00 | 512 |
| Large (l) | 1.00 | 1.00 | 512 |
| Extra-large (x) | 1.00 | 1.50 | 512 |

## Source-backed details for nano

The stage numbers below follow Wu et al., Figure 4. Saeheaw and Zhao,
Appendix B, Algorithm A1, independently describe the same joins and give
tensor sizes for a 640 × 640 input. Read A1 for the baseline. The subsequent
algorithms describe experimental replacements. [Wu et al.](https://doi.org/10.3390/bioengineering12020134)
and [Saeheaw and Zhao](https://doi.org/10.3390/buildings16091728).

| Join stage | Inputs, in the order reported by A1 | Following C3k2 output (channels × height × width) |
|---|---|---|
| 12 | Upsampled stage 10, backbone stage 6 | Stage 13 · 128 × 40 × 40 |
| 15 | Upsampled stage 13, backbone stage 4 | Stage 16 · 64 × 80 × 80 (P3) |
| 18 | Downsampled stage 16, neck stage 13 | Stage 19 · 128 × 40 × 40 (P4) |
| 21 | Downsampled stage 19, backbone stage 10 | Stage 22 · 256 × 20 × 20 (P5) |

Both upsampling steps double the spatial dimensions. Both downsampling steps
use a 3 × 3 convolution with stride 2. The head receives stages 16, 19 and
22 at strides 8, 16 and 32. A synthetic 640 × 640 shape test can therefore
expect 6,400, 1,600 and 400 prediction sites respectively. The interpolation
mode still needs a source. [Saeheaw and Zhao, Algorithm A1](https://doi.org/10.3390/buildings16091728)
and [Feng et al., baseline Figure 1](https://doi.org/10.3390/fire9110478).

**Scaling and rounding.** A1 supplies concrete nano widths, including 16 and
32 for the first two convolutions. An independent contributor can use the
published widths directly for a fixed nano graph. The scale presets and these
already-scaled widths cannot distinguish the channel divisor, rounding
direction, cap order, repeat minimum or half-way tie rule. A1 reports repeat
arguments for individual stages, but gives no rule for converting a family
repeat count into an instantiated nano block count. Verify those counts
separately before claiming parity. [Algorithm A1](https://doi.org/10.3390/buildings16091728)
and [Xu et al., Table 1](https://doi.org/10.3390/rs17203414).

**Attention.** A1 reports a 256-channel, 20 × 20 output at stage 10 with
expansion parameter 0.5. Meng et al., Figure 2a, shows one split path passing
through PSA blocks and the other reaching the concatenation directly. Neither
source specifies the nano head count, per-head query/key/value widths, or
positional projection. The outer expansion parameter alone cannot determine
those dimensions. [Algorithm A1](https://doi.org/10.3390/buildings16091728)
and [Meng et al., Figure 2a](https://doi.org/10.3390/s25175311).

**Detection head.** Feng et al., Figure 1, shows separate box and class towers
at each level, with two convolution stages drawn in each tower. At each site
the raw output has four distributions of `reg_max` logits and one logit per
class. Zhang et al. describe depthwise and pointwise convolution in the
baseline Detect module. Exact assignment of those operators to layers, tower
widths, bin count, grid origin and decoding order still need source support.
Keep the raw tower contract distinct from the decoded adapter contract below.
[Feng et al., baseline Figure 1](https://doi.org/10.3390/fire9110478)
and [Zhang et al., YOLO11n architecture](https://doi.org/10.1007/s44443-025-00116-0).

These additional papers publish their architectural descriptions under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The notes summarize
the authors' descriptions in new wording. The article licenses cover the
papers; code and checkpoint provenance require separate evidence.

## Independent implementation of nano

1. **Establish the graph.** Start with the baseline nano diagram in Wu et al.,
   Figure 4. Record a source for each stage, skip connection, feature-map size,
   channel count and repeat count. Apply the nano preset where the scaling rule
   is documented. Flag missing details for the maintainer before encoding them.
2. **Implement the family-specific blocks.** Use Meng et al., Figure 2, to
   identify the C3K2 and C2PSA paths. Verify branch widths, residual connections,
   attention dimensions and positional operations against approved sources.
   Reuse native convolution and bottleneck primitives where their contracts fit.
3. **Connect the prediction head.** Establish the nano prediction towers and
   their output layout independently. Evaluate LibreYOLO's existing
   [distribution projection](../yolo9/nn.py) and [box decoder](../yolo9/loss.py)
   against that design. The existing [port adapter](transforms.py) consumes
   decoded pixel-space xyxy boxes and class probabilities, with shape
   `(batch, 4 + classes, anchors)`.
4. **Verify the scratch model.** Instantiate it with random initialization.
   Test feature shapes at each detection level, a finite forward and backward
   pass, and decoded predictions through the existing adapter on synthetic
   images of different sizes. Record the implementation sources and any open
   decisions alongside the test results before registering the family.

## Paper references

All four articles are published under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Use the baseline YOLO11 material identified below; the papers also describe
task-specific systems and modified detectors.

- Wu et al. (2025), *Deep Learning-Assisted Diagnostic System: Apices and Odontogenic Sinus Floor Level Analysis in Dental Panoramic Radiographs*. Figure 4 shows the YOLO11n graph. [Article](https://doi.org/10.3390/bioengineering12020134)
- Meng et al. (2025), *Enhanced YOLOv11 Framework for Accurate Multi-Fault Detection in UAV Photovoltaic Inspection*. Figure 2 shows C2PSA and C3K2. [Article](https://doi.org/10.3390/s25175311)
- Xu et al. (2025), *SRTSOD-YOLO: Stronger Real-Time Small Object Detection Algorithm Based on Improved YOLO11 for UAV Imageries*. Table 1 lists all five baseline scale presets. [Article](https://doi.org/10.3390/rs17203414)
- Wei et al. (2026), *YOLO-PowerLite V2: An Enhanced Lightweight Detector for Real-Time Tiny Anomaly Identification on Overhead Transmission Lines in Complex Environments*. Table 1 corroborates all five presets. [Article](https://doi.org/10.3390/rs18121937)
