"""Tests for infrastructure setup and fixtures.

This module verifies that the test infrastructure is correctly configured
and that all fixtures work as expected.

Author: Refactored for Python 3.12+ best practices
Date: 2024-11-29
"""

import numpy as np
import pytest

from src.config import (
    DiceQuality,
    DistanceMetric,
    ImagingConstants,
    KMeansConfig,
    PostProcessConfig,
    SegmentationResult,
    TissueType,
)
from src.exceptions import (
    CardiacSegmentationError,
    ClusteringError,
    DataLoadError,
    DicomReadError,
    SegmentationError,
    ShapeMismatchError,
    ValidationError,
)
from src.types import BinaryMask, ImageStack, TimeCurves, TriggerTimes


# ============================================================================
# Exception Tests
# ============================================================================

class TestExceptions:
    """Test custom exception hierarchy."""

    def test_base_exception(self) -> None:
        """Test that CardiacSegmentationError can be raised."""
        with pytest.raises(CardiacSegmentationError):
            raise CardiacSegmentationError("Test error")

    def test_exception_hierarchy(self) -> None:
        """Test that all exceptions inherit from base class."""
        assert issubclass(DataLoadError, CardiacSegmentationError)
        assert issubclass(DicomReadError, DataLoadError)
        assert issubclass(ValidationError, CardiacSegmentationError)
        assert issubclass(ShapeMismatchError, ValidationError)
        assert issubclass(ClusteringError, CardiacSegmentationError)
        assert issubclass(SegmentationError, CardiacSegmentationError)

    def test_dicom_read_error_with_path(self) -> None:
        """Test DicomReadError with file path."""
        error = DicomReadError("Failed to read", file_path="/path/to/file.dcm")
        assert "/path/to/file.dcm" in str(error)

    def test_shape_mismatch_error_with_shapes(self) -> None:
        """Test ShapeMismatchError with shape information."""
        error = ShapeMismatchError(
            "Shape mismatch",
            expected_shape=(128, 128, 30),
            actual_shape=(64, 64, 30)
        )
        assert "(128, 128, 30)" in str(error)
        assert "(64, 64, 30)" in str(error)


# ============================================================================
# Config Tests
# ============================================================================

class TestEnums:
    """Test enum definitions."""

    def test_tissue_type_enum(self) -> None:
        """Test TissueType enum has all required members."""
        assert len(TissueType) == 4
        assert TissueType.BACKGROUND.value == "background"
        assert TissueType.LEFT_VENTRICLE.value == "lv"

    def test_tissue_type_colors(self) -> None:
        """Test TissueType color mapping."""
        lv_color = TissueType.LEFT_VENTRICLE.color_rgb
        assert len(lv_color) == 3
        assert all(0 <= c <= 1 for c in lv_color)

    def test_distance_metric_enum(self) -> None:
        """Test DistanceMetric enum."""
        assert DistanceMetric.EUCLIDEAN.scipy_name == "euclidean"
        assert DistanceMetric.CORRELATION.scipy_name == "correlation"

    def test_dice_quality_classification(self) -> None:
        """Test DiceQuality classification from scores."""
        assert DiceQuality.from_score(0.95) == DiceQuality.EXCELLENT
        assert DiceQuality.from_score(0.75) == DiceQuality.GOOD
        assert DiceQuality.from_score(0.60) == DiceQuality.MODERATE
        assert DiceQuality.from_score(0.40) == DiceQuality.POOR


class TestConstants:
    """Test imaging constants."""

    def test_constants_exist(self) -> None:
        """Test that all required constants are defined."""
        assert hasattr(ImagingConstants, "DEFAULT_PEAK_FRAME")
        assert hasattr(ImagingConstants, "MIN_TEMPORAL_FRAMES")
        assert hasattr(ImagingConstants, "N_CARDIAC_CLUSTERS")

    def test_constants_values(self) -> None:
        """Test that constants have reasonable values."""
        assert ImagingConstants.N_CARDIAC_CLUSTERS == 4
        assert ImagingConstants.MIN_TEMPORAL_FRAMES >= 1
        assert ImagingConstants.EPSILON > 0


class TestDataclasses:
    """Test configuration dataclasses."""

    def test_kmeans_config_defaults(self, default_kmeans_config: KMeansConfig) -> None:
        """Test KMeansConfig default values."""
        assert default_kmeans_config.n_clusters == 4
        assert default_kmeans_config.random_state == 42
        assert default_kmeans_config.metric == DistanceMetric.EUCLIDEAN

    def test_kmeans_config_frozen(self) -> None:
        """Test that KMeansConfig is immutable."""
        config = KMeansConfig()
        with pytest.raises(AttributeError):
            config.n_clusters = 5  # type: ignore

    def test_kmeans_config_validation(self) -> None:
        """Test KMeansConfig parameter validation."""
        with pytest.raises(ValueError, match="n_clusters must be >= 2"):
            KMeansConfig(n_clusters=1)

        with pytest.raises(ValueError, match="max_iter must be >= 1"):
            KMeansConfig(max_iter=0)

    def test_postprocess_config_defaults(
        self, default_postprocess_config: PostProcessConfig
    ) -> None:
        """Test PostProcessConfig default values."""
        assert default_postprocess_config.apply_morphology is True
        assert default_postprocess_config.erosion_kernel_size == 3
        assert default_postprocess_config.fill_holes is True

    def test_postprocess_config_validation(self) -> None:
        """Test PostProcessConfig parameter validation."""
        with pytest.raises(ValueError, match="erosion_kernel_size must be odd"):
            PostProcessConfig(erosion_kernel_size=4)

        with pytest.raises(ValueError, match="min_component_size must be >= 0"):
            PostProcessConfig(min_component_size=-1)

    def test_segmentation_result(
        self, sample_tissue_masks: dict[TissueType, BinaryMask]
    ) -> None:
        """Test SegmentationResult dataclass."""
        result = SegmentationResult(
            image_stack_shape=(128, 128, 30),
            cluster_labels=np.zeros((128, 128), dtype=np.int32),
            tissue_masks=sample_tissue_masks,
            dice_scores={"lv_vs_manual": 0.85, "myo_vs_manual": 0.75},
        )

        assert result.image_stack_shape == (128, 128, 30)
        assert "lv_vs_manual" in result.dice_scores
        assert result.get_average_dice() == pytest.approx(0.80)

    def test_segmentation_result_quality_summary(self) -> None:
        """Test SegmentationResult quality classification."""
        result = SegmentationResult(
            image_stack_shape=(128, 128, 30),
            cluster_labels=np.zeros((128, 128), dtype=np.int32),
            tissue_masks={},
            dice_scores={"test1": 0.95, "test2": 0.60},
        )

        quality = result.get_quality_summary()
        assert quality["test1"] == DiceQuality.EXCELLENT
        assert quality["test2"] == DiceQuality.MODERATE


# ============================================================================
# Fixture Tests
# ============================================================================

class TestFixtures:
    """Test that pytest fixtures work correctly."""

    def test_sample_image_stack_fixture(self, sample_image_stack: ImageStack) -> None:
        """Test sample_image_stack fixture."""
        assert isinstance(sample_image_stack, np.ndarray)
        assert sample_image_stack.ndim == 3
        assert sample_image_stack.shape == (128, 128, 30)
        assert sample_image_stack.dtype == np.float32
        assert 0 <= sample_image_stack.min() <= sample_image_stack.max() <= 1

    def test_sample_trigger_times_fixture(
        self, sample_trigger_times: TriggerTimes
    ) -> None:
        """Test sample_trigger_times fixture."""
        assert isinstance(sample_trigger_times, np.ndarray)
        assert sample_trigger_times.ndim == 1
        assert len(sample_trigger_times) == 30
        assert sample_trigger_times.dtype == np.float32
        assert np.all(sample_trigger_times[1:] > sample_trigger_times[:-1])

    def test_sample_time_curves_fixture(self, sample_time_curves: TimeCurves) -> None:
        """Test sample_time_curves fixture."""
        assert isinstance(sample_time_curves, np.ndarray)
        assert sample_time_curves.ndim == 2
        assert sample_time_curves.shape == (1000, 30)
        assert sample_time_curves.dtype == np.float32

    def test_sample_binary_mask_fixture(self, sample_binary_mask: BinaryMask) -> None:
        """Test sample_binary_mask fixture."""
        assert isinstance(sample_binary_mask, np.ndarray)
        assert sample_binary_mask.ndim == 2
        assert sample_binary_mask.shape == (128, 128)
        assert sample_binary_mask.dtype == bool
        assert sample_binary_mask.any()  # Should have some True values

    def test_sample_tissue_masks_fixture(
        self, sample_tissue_masks: dict[TissueType, BinaryMask]
    ) -> None:
        """Test sample_tissue_masks fixture."""
        assert len(sample_tissue_masks) == 4
        assert TissueType.LEFT_VENTRICLE in sample_tissue_masks

        # Check masks are non-overlapping
        all_masks = np.stack(list(sample_tissue_masks.values()))
        assert np.all(all_masks.sum(axis=0) == 1)  # Each pixel in exactly one mask

    def test_fast_kmeans_config_fixture(
        self, fast_kmeans_config: KMeansConfig
    ) -> None:
        """Test fast_kmeans_config fixture."""
        assert fast_kmeans_config.max_iter < 100
        assert fast_kmeans_config.n_init < 10
