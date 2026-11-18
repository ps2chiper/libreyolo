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
