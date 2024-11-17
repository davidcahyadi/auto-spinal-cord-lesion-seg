from pathlib import Path
from typing import List

from src.analyzer.image_classifier import ImageClassifier
from src.analyzer.lesion_segmenter import ImageSegmenter
from src.constant import MODEL_LABELS, MODEL_NAMES
from src.model.EfficientViT import EfficientVitMSRAWrapper
from src.model.STAMPUnet import STAMPUNet
from src.model.UNET import UNET
from src.utils.inference_dataset import InferenceDataset


class ImageProcessor:
    def __init__(self, weights_path: Path, format: str, prune: bool = False):
        self.format = format
        self.weights_path = weights_path
        self.models = {}
        self._load_models(prune)

    def _load_models(self, prune: bool = False):
        """Load all pre-trained models"""
        self.models[MODEL_NAMES.ANATOMY] = EfficientVitMSRAWrapper.load_from_checkpoint(
            self.weights_path / f"{MODEL_NAMES.ANATOMY}.ckpt"
        ).eval()
        self.models[MODEL_NAMES.LOCATION_VIEW] = (
            EfficientVitMSRAWrapper.load_from_checkpoint(
                self.weights_path / f"{MODEL_NAMES.LOCATION_VIEW}.ckpt"
            )
        ).eval()
        self.models[MODEL_NAMES.SEQUENCE] = (
            EfficientVitMSRAWrapper.load_from_checkpoint(
                self.weights_path / f"{MODEL_NAMES.SEQUENCE}.ckpt"
            )
        ).eval()
        self.models[MODEL_NAMES.SPINAL_CORD] = (
            EfficientVitMSRAWrapper.load_from_checkpoint(
                self.weights_path / f"{MODEL_NAMES.SPINAL_CORD}.ckpt"
            )
        ).eval()

        if prune:
            self.models[MODEL_NAMES.LESION_SEGMENTATION] = (
                STAMPUNet.load_from_checkpoint(
                    self.weights_path / f"{MODEL_NAMES.PRUNED_LESION_SEGMENTATION}.ckpt"
                )
            ).eval()
        else:
            self.models[MODEL_NAMES.LESION_SEGMENTATION] = UNET.load_from_checkpoint(
                self.weights_path / f"{MODEL_NAMES.LESION_SEGMENTATION}.ckpt"
            ).eval()

    def process_directory(self, directory_path: Path, output_dir: Path):
        """Process all series directories in the main folder"""
        images = list(str(path) for path in directory_path.rglob(f"*.{self.format}"))
        if not images:
            print(f"No valid images found in {directory_path}")
            return

        anatomy_images = self._process_anatomy(images)
        location_view_images = self._process_location_view(anatomy_images)
        sequence_images = self._process_sequence(location_view_images)
        spinal_cord_images = self._process_spinal_cord(sequence_images)
        self._process_lesion_segmentation(spinal_cord_images, output_dir)

    def _process_anatomy(self, images: List[str]) -> List[str]:
        """Process anatomy images using a pre-trained model"""
        print("Filter based on spine anatomy series ")
        classifier = ImageClassifier(
            self.models[MODEL_NAMES.ANATOMY],
            device="cpu",
            batch_size=16,
            aggregate_by="majority",
        )
        dataset = InferenceDataset(images, size=224)
        filtered_anatomy, avg_time = classifier.classify(
            dataset,
            selected_label=MODEL_LABELS.ANATOMY,
        )
        print(
            f"Done in {avg_time:.2f}s, removed {len(images) - len(filtered_anatomy)} images, selected {len(filtered_anatomy)} images"
        )
        return filtered_anatomy

    def _process_location_view(self, images: List[str]):
        """Process location view images using a pre-trained model"""
        print("Filter based on cervical sagittal view series ")
        classifier = ImageClassifier(
            self.models[MODEL_NAMES.LOCATION_VIEW],
            device="cpu",
            batch_size=16,
            aggregate_by="majority",
        )
        dataset = InferenceDataset(images)
        filtered_location_view, avg_time = classifier.classify(
            dataset,
            selected_label=MODEL_LABELS.LOCATION_VIEW,
        )
        print(
            f"Done in {avg_time:.2f}s, removed {len(images) - len(filtered_location_view)} images, selected {len(filtered_location_view)} images"
        )
        return filtered_location_view

    def _process_sequence(self, images: List[str]):
        """Process sequence images using a pre-trained model"""
        print("Filter based on T2-weighted series ")
        classifier = ImageClassifier(
            self.models[MODEL_NAMES.SEQUENCE],
            device="cpu",
            batch_size=16,
            aggregate_by="majority",
        )
        dataset = InferenceDataset(images)
        filtered_sequence, avg_time = classifier.classify(
            dataset,
            selected_label=MODEL_LABELS.SEQUENCE,
        )
        print(
            f"Done in {avg_time:.2f}s, removed {len(images) - len(filtered_sequence)} images, selected {len(filtered_sequence)} images"
        )
        return filtered_sequence

    def _process_spinal_cord(self, images: List[str]):
        """Process spinal cord images using a pre-trained model"""
        print("Filter based on spinal cord slices ")
        classifier = ImageClassifier(
            self.models[MODEL_NAMES.SPINAL_CORD],
            device="cpu",
            batch_size=16,
            aggregate_by="majority",
        )
        dataset = InferenceDataset(images)
        filtered_spinal_cord, total_time = classifier.classify(
            dataset,
            selected_label=MODEL_LABELS.SPINAL_CORD,
            group_level=0,
            min_series_length=1,
        )
        print(
            f"Done in {total_time:.4f}s, removed {len(images) - len(filtered_spinal_cord)} images, selected {len(filtered_spinal_cord)} images"
        )
        return filtered_spinal_cord

    def _process_lesion_segmentation(self, images: List[str], output_dir: Path):
        """Process lesion segmentation images"""
        print("Segmenting lesion images ", end="")
        segmenter = ImageSegmenter(
            self.models[MODEL_NAMES.LESION_SEGMENTATION],
            device="cpu",
            batch_size=16,
        )
        dataset = InferenceDataset(images, size=320)
        total_time = segmenter.segment(dataset, output_dir)
        print(f"Done in {total_time:.4f}s")

    def _get_series_directories(self, main_path: Path):
        """Get all series directories from the main folder"""
        return [d for d in main_path.iterdir() if d.is_dir()]
