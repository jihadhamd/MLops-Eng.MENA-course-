"""Tests for feature engineering and validation."""

import pytest

from prodml.features import validate_features


class TestValidateFeatures:
    """Edge cases for validate_features()."""

    def test_valid_features_pass(self):
        assert validate_features({"PU_DO": "74_41", "trip_distance": 5.0}) is True

    def test_zero_distance_fails(self):
        assert validate_features({"PU_DO": "74_41", "trip_distance": 0.0}) is False

    def test_negative_distance_fails(self):
        assert validate_features({"PU_DO": "74_41", "trip_distance": -5.0}) is False

    def test_distance_too_large_fails(self):
        assert validate_features({"PU_DO": "74_41", "trip_distance": 250.0}) is False

    def test_missing_pu_do_fails(self):
        """Missing category - no PU_DO key at all."""
        assert validate_features({"trip_distance": 5.0}) is False

    def test_empty_pu_do_fails(self):
        """PU_DO present but empty string."""
        assert validate_features({"PU_DO": "", "trip_distance": 5.0}) is False

    def test_unseen_pu_do_pair_still_valid_shape(self):
        """An unseen PU_DO pair (e.g. a route not in training data) is still
        structurally valid - the vectorizer handles unknown categories at
        prediction time, not validate_features(). This just confirms
        validation doesn't reject it prematurely based on the string value."""
        assert validate_features({"PU_DO": "999_999", "trip_distance": 5.0}) is True

    @pytest.mark.parametrize("distance", [0.01, 50.0, 199.99])
    def test_boundary_values_pass(self, distance):
        assert validate_features({"PU_DO": "1_1", "trip_distance": distance}) is True

    @pytest.mark.parametrize("distance", [0, -1, 201, 500])
    def test_boundary_values_fail(self, distance):
        assert validate_features({"PU_DO": "1_1", "trip_distance": distance}) is False
