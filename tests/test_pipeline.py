import numpy as np
import pytest

from src.classical import mean_embeddings, vectorizer
from src.common import make_split, scores, tokenize, validate_split


def test_split_is_reproducible_complete_disjoint_and_stratified():
    labels = np.asarray(["sports"] * 750 + ["politics"] * 125 + ["business"] * 125)
    split = make_split(labels)
    assert split == make_split(labels)
    validate_split(split, 1000)
    assert [len(v) for v in split.values()] == [800, 100, 100]
    for indices in split.values():
        assert abs(np.mean(labels[indices] == "sports") - 0.75) < 0.02
    broken = {k: list(v) for k, v in split.items()}
    broken["test"][0] = broken["train"][0]
    with pytest.raises(ValueError):
        validate_split(broken, 1000)


def test_bow_counts_and_heldout_words_never_enter_vocabulary():
    for kind, expected in [("binary", [1, 1]), ("frequency", [2, 1])]:
        vec = vectorizer(kind)
        train = vec.fit_transform(["apple apple banana"])
        assert train.toarray()[0].tolist() == expected
        assert vec.transform(["heldouttoken"]).nnz == 0
        assert "heldouttoken" not in vec.vocabulary_
    tfidf = vectorizer("tfidf")
    tfidf.fit(["apple apple banana", "banana"])
    assert tfidf.idf_[tfidf.vocabulary_["apple"]] > tfidf.idf_[tfidf.vocabulary_["banana"]]
    before = tfidf.idf_.copy()
    tfidf.transform(["apple heldouttoken"])
    np.testing.assert_array_equal(before, tfidf.idf_)


def test_embedding_average_counts_repetitions_skips_oov_and_handles_empty():
    vectors = {"apple": np.array([2, 0]), "banana": np.array([0, 3])}
    x, stats = mean_embeddings([["apple", "apple", "banana", "missing"], ["missing"], []], vectors, dim=2)
    np.testing.assert_allclose(x, [[4 / 3, 1], [0, 0], [0, 0]])
    assert stats["coverage"] == 3 / 5
    assert stats["empty_documents"] == 2


def test_metrics_include_unpredicted_minority_classes():
    result = scores(["a", "a", "b"], ["a", "a", "a"], ["a", "b"])
    assert result["accuracy"] == pytest.approx(2 / 3)
    assert result["macro_f1"] == pytest.approx(0.4)
    assert result["confusion_matrix"] == [[2, 0], [1, 0]]


def test_tokenizer_is_shared_and_lowercases():
    assert tokenize("Apple, DON'T! 123 banana") == ["apple", "don't", "banana"]
