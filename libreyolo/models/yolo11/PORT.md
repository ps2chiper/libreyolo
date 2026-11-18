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
