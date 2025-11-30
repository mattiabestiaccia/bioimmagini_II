"""
Custom exceptions for noise analysis module.

This module defines exception classes used throughout the noise analysis package.
"""


class NoiseAnalysisError(Exception):
    """Base exception for noise analysis errors."""

    pass


class DataLoadError(NoiseAnalysisError):
    """Exception raised when data loading fails."""

    def __init__(self, filepath: str, reason: str):
        self.filepath = filepath
        self.reason = reason
        super().__init__(f"Failed to load data from {filepath}: {reason}")


class DICOMReadError(DataLoadError):
    """Exception raised when DICOM file reading fails."""

    def __init__(self, filepath: str, reason: str = "Invalid DICOM file"):
        super().__init__(filepath, reason)


class ImageProcessingError(NoiseAnalysisError):
    """Exception raised during image processing operations."""

    pass


class SDMapComputationError(ImageProcessingError):
    """Exception raised when SD map computation fails."""

    def __init__(self, reason: str):
        super().__init__(f"SD map computation failed: {reason}")


class ROIError(NoiseAnalysisError):
    """Exception raised for ROI-related errors."""

    pass


class InvalidROIError(ROIError):
    """Exception raised when ROI parameters are invalid."""

    def __init__(self, reason: str):
        super().__init__(f"Invalid ROI: {reason}")


class ROIOutOfBoundsError(ROIError):
    """Exception raised when ROI is outside image bounds."""

    def __init__(self, roi_coords: tuple, image_shape: tuple):
        self.roi_coords = roi_coords
        self.image_shape = image_shape
        super().__init__(
            f"ROI {roi_coords} is outside image bounds {image_shape}"
        )


class EstimationError(NoiseAnalysisError):
    """Exception raised during statistical estimation."""

    pass


class HistogramEstimationError(EstimationError):
    """Exception raised when histogram-based estimation fails."""

    def __init__(self, reason: str):
        super().__init__(f"Histogram estimation failed: {reason}")


class ValidationError(NoiseAnalysisError):
    """Exception raised during data validation."""

    pass


class EmptyDataError(ValidationError):
    """Exception raised when data is empty or invalid."""

    def __init__(self, data_name: str):
        super().__init__(f"{data_name} is empty or contains no valid data")


class InvalidParameterError(ValidationError):
    """Exception raised when function parameters are invalid."""

    def __init__(self, param_name: str, value: any, reason: str):
        self.param_name = param_name
        self.value = value
        super().__init__(
            f"Invalid parameter '{param_name}' = {value}: {reason}"
        )


class ConfigurationError(NoiseAnalysisError):
    """Exception raised for configuration-related errors."""

    pass


class MissingConfigError(ConfigurationError):
    """Exception raised when required configuration is missing."""

    def __init__(self, config_key: str):
        super().__init__(f"Required configuration missing: {config_key}")


class VisualizationError(NoiseAnalysisError):
    """Exception raised during visualization operations."""

    pass


class PlotCreationError(VisualizationError):
    """Exception raised when plot creation fails."""

    def __init__(self, plot_type: str, reason: str):
        super().__init__(f"Failed to create {plot_type} plot: {reason}")


class SaveError(NoiseAnalysisError):
    """Exception raised when saving results fails."""

    def __init__(self, filepath: str, reason: str):
        super().__init__(f"Failed to save to {filepath}: {reason}")
