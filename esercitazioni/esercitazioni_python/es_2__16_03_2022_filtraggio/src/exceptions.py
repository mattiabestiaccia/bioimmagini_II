"""
Custom exceptions for 3D CT filtering module.

This module defines exception classes used throughout the filtering package.
"""


class FilteringError(Exception):
    """Base exception for filtering errors."""

    pass


class DataLoadError(FilteringError):
    """Exception raised when data loading fails."""

    def __init__(self, filepath: str, reason: str):
        self.filepath = filepath
        self.reason = reason
        super().__init__(f"Failed to load data from {filepath}: {reason}")


class DICOMReadError(DataLoadError):
    """Exception raised when DICOM file reading fails."""

    def __init__(self, filepath: str, reason: str = "Invalid DICOM file"):
        super().__init__(filepath, reason)


class VolumeProcessingError(FilteringError):
    """Exception raised during volume processing operations."""

    pass


class IsotropyError(VolumeProcessingError):
    """Exception raised when isotropic conversion fails."""

    def __init__(self, reason: str):
        super().__init__(f"Isotropic conversion failed: {reason}")


class InterpolationError(VolumeProcessingError):
    """Exception raised when interpolation fails."""

    def __init__(self, reason: str):
        super().__init__(f"Interpolation failed: {reason}")


class FilterError(FilteringError):
    """Exception raised during filtering operations."""

    pass


class FilterParameterError(FilterError):
    """Exception raised when filter parameters are invalid."""

    def __init__(self, param_name: str, value: any, reason: str):
        self.param_name = param_name
        self.value = value
        super().__init__(f"Invalid filter parameter '{param_name}' = {value}: {reason}")


class NoiseEstimationError(FilterError):
    """Exception raised when noise estimation fails."""

    def __init__(self, reason: str):
        super().__init__(f"Noise estimation failed: {reason}")


class ROIError(FilteringError):
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
        super().__init__(f"ROI {roi_coords} is outside image bounds {image_shape}")


class MetricsError(FilteringError):
    """Exception raised during metrics calculation."""

    pass


class SNRCalculationError(MetricsError):
    """Exception raised when SNR calculation fails."""

    def __init__(self, reason: str):
        super().__init__(f"SNR calculation failed: {reason}")


class SharpnessCalculationError(MetricsError):
    """Exception raised when sharpness calculation fails."""

    def __init__(self, reason: str):
        super().__init__(f"Sharpness calculation failed: {reason}")


class ProfileExtractionError(MetricsError):
    """Exception raised when profile extraction fails."""

    def __init__(self, reason: str):
        super().__init__(f"Profile extraction failed: {reason}")


class ValidationError(FilteringError):
    """Exception raised during data validation."""

    pass


class EmptyDataError(ValidationError):
    """Exception raised when data is empty or invalid."""

    def __init__(self, data_name: str):
        super().__init__(f"{data_name} is empty or contains no valid data")


class InvalidDimensionsError(ValidationError):
    """Exception raised when data dimensions are invalid."""

    def __init__(self, expected: tuple, actual: tuple):
        super().__init__(f"Invalid dimensions: expected {expected}, got {actual}")


class InvalidParameterError(ValidationError):
    """Exception raised when function parameters are invalid."""

    def __init__(self, param_name: str, value: any, reason: str):
        self.param_name = param_name
        self.value = value
        super().__init__(f"Invalid parameter '{param_name}' = {value}: {reason}")


class ConfigurationError(FilteringError):
    """Exception raised for configuration-related errors."""

    pass


class MissingConfigError(ConfigurationError):
    """Exception raised when required configuration is missing."""

    def __init__(self, config_key: str):
        super().__init__(f"Required configuration missing: {config_key}")


class VisualizationError(FilteringError):
    """Exception raised during visualization operations."""

    pass


class PlotCreationError(VisualizationError):
    """Exception raised when plot creation fails."""

    def __init__(self, plot_type: str, reason: str):
        super().__init__(f"Failed to create {plot_type} plot: {reason}")


class SaveError(FilteringError):
    """Exception raised when saving results fails."""

    def __init__(self, filepath: str, reason: str):
        super().__init__(f"Failed to save to {filepath}: {reason}")
