from ...preprocess.letterbox import LETTERBOX_CENTER
from ...training.trainer import BaseTrainer
from ..yolo9.loss import YOLO9Loss
from ..yolo9.trainer import DenseDetectionRecipe
from ..yolo9.validation_loss import DenseDetectionValidationLoss
from .config import ExperimentalNanoConfig


class ExperimentalNanoTrainer(DenseDetectionRecipe, BaseTrainer):
    """Train the experimental head with the native task-aligned box/class/DFL loss."""

    artifact_model_families = ("yolo11_experimental",)

    @classmethod
    def _config_class(cls):
        return ExperimentalNanoConfig

    def get_model_family(self):
        return "yolo11_experimental"

    def get_model_tag(self):
        return "YOLO11n-experimental"

    def _resolved_letterbox_pad(self):
        if self.config.letterbox_pad != LETTERBOX_CENTER:
            raise ValueError("The experimental nano preset requires centered letterboxing")
        return LETTERBOX_CENTER

    def on_setup(self):
        head = self.wrapper_model.model.head
        self.loss = YOLO9Loss(
            num_classes=head.nc, reg_max=head.reg_max,
            strides=list(self.wrapper_model.model.decoder.strides),
            image_size=None, device=self.device,
        )

    def on_forward(self, imgs, targets, polygons=None):
        raw = self.model(imgs, return_raw=True)
        self.loss.update_anchors([imgs.shape[-1], imgs.shape[-2]])
        return self.loss(raw, targets)

    def validate_validation_loss_config(self):
        """The validation adapter supports the preset's detection task."""

    def build_validation_loss_adapter(self, model):
        return DenseDetectionValidationLoss(
            num_classes=model.head.nc, reg_max=model.head.reg_max,
            strides=model.decoder.strides, device=next(model.parameters()).device,
            max_labels=self.config.max_labels,
        )

    def _checkpoint_extra_metadata(self):
        return {**super()._checkpoint_extra_metadata(), **self.wrapper_model._save_extra_metadata()}
