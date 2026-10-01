"""Ground truth is valid only for an unchanged historical input."""

from pages_src.predict import _matches_loaded_student


def test_student_ground_truth_is_invalidated_by_edit():
    loaded = {"Age at enrollment": 18, "Tuition fees up to date": 1}

    assert _matches_loaded_student(loaded.copy(), loaded)
    assert not _matches_loaded_student({**loaded, "Age at enrollment": 19}, loaded)
    assert not _matches_loaded_student(loaded, {})
