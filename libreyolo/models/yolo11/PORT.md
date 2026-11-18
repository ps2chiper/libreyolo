# YOLO11 detection port

This branch implements the image and detection boundary using LibreYOLO's native
preprocessing, box decoding, letterbox geometry and class-aware postprocessing.
Call `YOLO11nDecoder()([p3, p4, p5])` on raw maps in stride order (8, 16, 32).
Each BCHW map contains 64 box logits followed by one logit per class. Pass the
result directly to `restore_detections` with the original image sizes and the
centered letterbox canvas size. Synthetic tests exercise this path.
[Decoder](decoder.py) and [image adapters](transforms.py).

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
the decoder's three raw inputs. Factory registration, training, validation and
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
widths, bin count, grid origin and decoding order are unspecified in these papers.
Keep the raw tower contract distinct from the decoded adapter contract below.
[Feng et al., baseline Figure 1](https://doi.org/10.3390/fire9110478)
and [Zhang et al., YOLO11n architecture](https://doi.org/10.1007/s44443-025-00116-0).

These additional papers publish their architectural descriptions under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The notes summarize
the authors' descriptions in new wording. The article licenses cover the
papers; code and checkpoint provenance require separate evidence.

## Implementation evidence for the nano head and decoder

**Decoder.** Tencent's ncnn YOLO11 example carries a file-level BSD-3-Clause
notice. Its CPU postprocessor consumes a raw row per prediction site, in
row-major grid order. The first 64 logits are four contiguous groups of 16,
ordered left, top, right, bottom. For each group it takes a softmax over bins
0 through 15 and computes the probability-weighted mean bin index. At grid
column x and row y, the center in grid units is (x + 0.5, y + 0.5).
Subtract the left and top means and add the right and bottom means, then
multiply by the level stride to get pixel-space xyxy on the input canvas.
Class logits follow the 64 box logits. The example selects the highest class
and applies sigmoid, without an objectness multiplier. LibreYOLO's adapter
needs all class probabilities, so retain the per-class sigmoid scores before
filtering. [ncnn decoder](https://github.com/Tencent/ncnn/blob/71f731859876c6da4195949a85a9c0b188304705/examples/yolo11.cpp#L186-L281)
and [license](https://github.com/Tencent/ncnn/blob/71f731859876c6da4195949a85a9c0b188304705/LICENSE.txt).

Hailo's nano-specific COCO configuration corroborates 16 regression bins,
80 classes, separate box and class outputs at strides 8, 16 and 32, and
sigmoid on all three class outputs. Its configuration repository is MIT.
Its network manifest identifies the upstream model as AGPL-3.0. These
references establish the output contract for an independently trained model.
[Nano decoder configuration](https://github.com/hailo-ai/hailo_model_zoo/blob/888f366800dcbd5be7c188a04e56480bba75e9a7/hailo_model_zoo/cfg/postprocess_config/yolov11n_nms_config.json),
[activation configuration](https://github.com/hailo-ai/hailo_model_zoo/blob/888f366800dcbd5be7c188a04e56480bba75e9a7/hailo_model_zoo/cfg/alls/generic/yolov11n.alls),
[network manifest](https://github.com/hailo-ai/hailo_model_zoo/blob/888f366800dcbd5be7c188a04e56480bba75e9a7/hailo_model_zoo/cfg/networks/yolov11n.yaml)
and [license](https://github.com/hailo-ai/hailo_model_zoo/blob/888f366800dcbd5be7c188a04e56480bba75e9a7/LICENSE).

A synthetic decoder check follows from those contracts. At stride 8, column
10 and row 20, equal logits within each group give four mean distances of
7.5 cells. The pre-clipping box is (24, 104, 144, 224) in pixels. A zero
class logit gives probability 0.5. To detect swapped sides, also test four
distributions concentrated at distinct bins.

**Attention and detection tower observations.** TensorRTx has an MIT repository
license and a concrete nano builder. Its YOLO11 instructions identify a
specific Ultralytics release as training code, and its builders include
Python-style expressions alongside the C++ graph. This establishes the
behavior of that port. I could not verify independent authorship of the
corresponding graph logic under LibreYOLO's contribution policy. The
following observations are for provenance review before implementation.
[TensorRTx license](https://github.com/wang-xinyu/tensorrtx/blob/89db56262a761f6ac63544634757cd949b756ee3/LICENSE),
[YOLO11 instructions](https://github.com/wang-xinyu/tensorrtx/blob/89db56262a761f6ac63544634757cd949b756ee3/yolo11/readme.md)
and [LibreYOLO contribution policy](../../../CONTRIBUTING.md).

- For nano, its 256-channel stage sends 128 channels through attention.
  Two heads each use 32 query, 32 key and 64 value channels. Its positional
  path applies a 3 × 3 depthwise convolution to the spatial value tensor
  before addition and a 1 × 1 projection. [Nano preset](https://github.com/wang-xinyu/tensorrtx/blob/89db56262a761f6ac63544634757cd949b756ee3/yolo11/yolo11_det.cpp#L127-L140),
  [stage](https://github.com/wang-xinyu/tensorrtx/blob/89db56262a761f6ac63544634757cd949b756ee3/yolo11/src/model.cpp#L186-L191)
  and [attention builder](https://github.com/wang-xinyu/tensorrtx/blob/89db56262a761f6ac63544634757cd949b756ee3/yolo11/src/block.cpp#L293-L404).
- For the 80-class nano detector, its box tower has two 3 × 3
  convolution/BN/SiLU stages of width 64 and a 1 × 1 output of width 64.
  Its class tower uses two 3 × 3 depthwise and 1 × 1 pointwise pairs, with
  pointwise outputs of width 80, followed by a 1 × 1 class-logit output.
  The builder repeats the tower pattern at all three scales. The COCO class
  width is specific to that configuration. [Detection builder](https://github.com/wang-xinyu/tensorrtx/blob/89db56262a761f6ac63544634757cd949b756ee3/yolo11/src/model.cpp#L238-L315).

### TensorRTx source history and port decision

Keep both towers pending independent implementation. The TensorRTx root carried
MIT at both YOLO11 introductions. The original changes contain direct textual
links to AGPL-licensed source in the affected functions. Under LibreYOLO's
provenance policy, TensorRTx therefore cannot establish an independent source
for porting either tower. The evidence below concerns these particular files.
[2024 license](https://github.com/wang-xinyu/tensorrtx/blob/4f79443dd2e23547457063d614e8eda868667cf7/LICENSE),
[2025 license](https://github.com/wang-xinyu/tensorrtx/blob/de7ad561bc23c9d1521337e2b9f330410b77f17e/LICENSE)
and [contribution policy](../../../CONTRIBUTING.md).

- **Attention predates YOLO11.** The earliest matching TensorRTx PR commit
  located was mpj1234's YOLOv10 TensorRT 10 implementation on July 24, 2024
  (`070c81a`), followed by the TensorRT 8 version on July 26 (`1b86af0`).
  The October YOLO11 TensorRT 8 attention function retains the July TensorRT 8
  function body, including its comments. Two of those comments reproduce
  Python expressions from THU-MIG's YOLOv10 v1.1 attention implementation:
  the positional convolution declaration and the value aggregation plus
  positional addition. THU-MIG's license at that revision is AGPL-3.0.
  This traces the inner attention function; it does not establish the origin
  of every line of the enclosing C2PSA block.
  [July 24 commit](https://github.com/wang-xinyu/tensorrtx/commit/070c81ae9ecc5e8b47aeda745936f6cc99de7a42),
  [July 26 commit](https://github.com/wang-xinyu/tensorrtx/commit/1b86af028da36ccf684053a9cdde5f532e82a013),
  [original YOLO11 attention](https://github.com/wang-xinyu/tensorrtx/blob/4f79443dd2e23547457063d614e8eda868667cf7/yolo11/src/block.cpp#L281-L344),
  [THU-MIG attention](https://github.com/THU-MIG/yolov10/blob/799ff3be47d21173bcf29b351820d4b8e955e0fe/ultralytics/nn/modules/block.py#L783-L796)
  and [historical license](https://github.com/THU-MIG/yolov10/blob/799ff3be47d21173bcf29b351820d4b8e955e0fe/LICENSE).

- **YOLO11 detection towers first appear on October 11, 2024.** In PR
  #1584, mpj1234's first commit (`e78ab62`) added the TensorRT 8 builders.
  The second (`9ca5b46`) added the training-code link. The merged commit was
  `4f79443`. The comment directly above the C++ box- and class-width
  calculations matches the complete Python assignment in Ultralytics v8.3.0
  Detect, including its trailing comment. The original README names that
  release, whose license was AGPL-3.0. The matched text provides stronger
  evidence of source contact than checkpoint compatibility alone.
  [Original PR and commits](https://github.com/wang-xinyu/tensorrtx/pull/1584/commits),
  [merged detection builder](https://github.com/wang-xinyu/tensorrtx/blob/4f79443dd2e23547457063d614e8eda868667cf7/yolo11/src/model.cpp#L237-L262),
  [original README](https://github.com/wang-xinyu/tensorrtx/blob/4f79443dd2e23547457063d614e8eda868667cf7/yolo11/readme.md),
  [upstream Detect](https://github.com/ultralytics/ultralytics/blob/6e43d1e1e5db72afbf686dee6745669bcb124b0a/ultralytics/nn/modules/head.py#L40)
  and [historical license](https://github.com/ultralytics/ultralytics/blob/6e43d1e1e5db72afbf686dee6745669bcb124b0a/LICENSE).

- **The later TensorRT 10 port preserves both matches.** PR #1681 introduced
  the port in `2946b95` on December 20, 2025 (UTC), followed by formatting
  in `8306147` and the merged commit `de7ad56`. Its author names Ultralytics
  v8.3.238. Neither YOLO11 PR records an independent implementation process
  or a separate grant for the matched material. The public history establishes
  source contact and reuse across ports. It cannot determine how the author
  wrote each remaining TensorRT operation.
  [PR #1681](https://github.com/wang-xinyu/tensorrtx/pull/1681),
  [attention at merge](https://github.com/wang-xinyu/tensorrtx/blob/de7ad561bc23c9d1521337e2b9f330410b77f17e/yolo11/src/block.cpp#L337-L355)
  and [detection at merge](https://github.com/wang-xinyu/tensorrtx/blob/de7ad561bc23c9d1521337e2b9f330410b77f17e/yolo11/src/model.cpp#L238-L262).

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
   their output layout independently. The [nano decoder](decoder.py) already
   reuses the [native LibreYOLO decoding implementation](../yolo9/nn.py).
   The existing [port adapter](transforms.py) consumes
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
