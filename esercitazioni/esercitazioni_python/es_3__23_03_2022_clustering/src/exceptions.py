"""Custom exception hierarchy for cardiac segmentation pipeline.

This module defines a hierarchy of exceptions used throughout the cardiac
MRI segmentation pipeline, providing clear error types for different failure
modes.

Author: Refactored for Python 3.12+ best practices
Date: 2024-11-29
"""


class CardiacSegmentationError(Exception):
    """Base exception for all cardiac segmentation errors.

    All custom exceptions in this module inherit from this base class,
    allowing for catch-all error handling when needed.
    """


class DataLoadError(CardiacSegmentationError):
    """Exception raised when data loading fails.

    This includes failures in reading files, parsing data formats,
    or accessing required resources.
    """


class DicomReadError(DataLoadError):
    """Exception raised when DICOM file reading fails.

    Raised when DICOM files cannot be read, parsed, or when required
    DICOM tags are missing or invalid.

    Attributes:
        file_path: Path to the DICOM file that failed to read
        original_error: The underlying exception that caused the failure
    """

    def __init__(
        self,
        message: str,
        file_path: str | None = None,
        original_error: Exception | None = None,
    ) -> None:
        self.file_path = file_path
        self.original_error = original_error
        super().__init__(message)

    def __str__(self) -> str:
        msg = super().__str__()
        if self.file_path:
            msg = f"{msg} (file: {self.file_path})"
        if self.original_error:
            msg = f"{msg} - Original error: {self.original_error}"
        return msg


class ValidationError(CardiacSegmentationError):
    """Exception raised when data validation fails.

    Raised when input data does not meet expected requirements,
    such as incorrect shapes, invalid value ranges, or missing
    required properties.
    """


class ShapeMismatchError(ValidationError):
    """Exception raised when array shapes are incompatible.

    Raised when operations require arrays of specific shapes but
    receive incompatible dimensions.

    Attributes:
        expected_shape: The expected array shape
        actual_shape: The actual array shape received
    """

    def __init__(
        self,
        message: str,
        expected_shape: tuple[int, ...] | None = None,
        actual_shape: tuple[int, ...] | None = None,
    ) -> None:
        self.expected_shape = expected_shape
        self.actual_shape = actual_shape
        super().__init__(message)

    def __str__(self) -> str:
        msg = super().__str__()
        if self.expected_shape and self.actual_shape:
            msg = f"{msg} (expected: {self.expected_shape}, got: {self.actual_shape})"
        return msg


class InvalidClusterCountError(ValidationError):
    """Exception raised when cluster count is invalid.

    Raised when the number of clusters is less than the minimum required
    for the operation.

    Attributes:
        n_clusters: The invalid number of clusters provided
        min_required: The minimum number of clusters required
    """

    def __init__(
        self,
        message: str,
        n_clusters: int | None = None,
        min_required: int | None = None,
    ) -> None:
        self.n_clusters = n_clusters
        self.min_required = min_required
        super().__init__(message)

    def __str__(self) -> str:
        msg = super().__str__()
        if self.n_clusters is not None and self.min_required is not None:
            msg = f"{msg} (got: {self.n_clusters}, required: >= {self.min_required})"
        return msg


class ClusteringError(CardiacSegmentationError):
    """Exception raised when clustering operations fail.

    Raised when K-means or other clustering algorithms encounter
    errors during execution or produce invalid results.
    """


class SegmentationError(CardiacSegmentationError):
    """Exception raised when segmentation operations fail.

    Raised when segmentation algorithms fail to produce valid output,
    such as when post-processing steps fail or segmentation quality
    is below acceptable thresholds.
    """
