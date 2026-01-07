"""Simple test to verify scribble generation works."""

import numpy as np

from mask_to_scribble import generate_scribble


def test_generate_scribble_basic() -> None:
    """Test basic scribble generation from a simple square mask."""
    # Create a simple square mask
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[30:70, 30:70] = 255

    # Generate scribble
    scribble = generate_scribble(mask)

    # Check output shape and type
    assert scribble.shape == mask.shape
    assert scribble.dtype == np.uint8

    # Scribble should have some non-zero pixels
    assert np.sum(scribble > 0) > 0

    # Scribble should be smaller than mask
    assert np.sum(scribble > 0) < np.sum(mask > 0)

    print("✓ Basic scribble generation works")


def test_generate_scribble_soft_labels() -> None:
    """Test soft label generation."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[30:70, 30:70] = 255

    # Generate soft labels
    scribble = generate_scribble(mask, use_soft_labels=True)

    # Check output type
    assert scribble.dtype == np.float32

    # Values should be in [0, 1]
    assert scribble.min() >= 0
    assert scribble.max() <= 1

    # Should have some non-zero values
    assert np.sum(scribble > 0) > 0

    print("✓ Soft label generation works")


def test_generate_scribble_empty_mask() -> None:
    """Test that empty mask produces empty scribble."""
    mask = np.zeros((100, 100), dtype=np.uint8)

    scribble = generate_scribble(mask)

    assert np.sum(scribble) == 0
    assert scribble.shape == mask.shape

    print("✓ Empty mask handling works")


def test_determinism() -> None:
    """Test that same mask produces same scribble (deterministic)."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[30:70, 30:70] = 255

    scribble1 = generate_scribble(mask)
    scribble2 = generate_scribble(mask)

    # Should be identical
    assert np.array_equal(scribble1, scribble2)

    print("✓ Determinism verified")


if __name__ == "__main__":
    test_generate_scribble_basic()
    test_generate_scribble_soft_labels()
    test_generate_scribble_empty_mask()
    test_determinism()
    print("\n✅ All tests passed!")
