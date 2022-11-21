import confirm.imprint.checkpoint as checkpoint


def test_exponential_27():
    delete_idxs = checkpoint.exponential_delete(27)
    assert delete_idxs == list(range(1, 10)) + list(range(11, 17))


def test_exponential_213():
    delete_idxs = checkpoint.exponential_delete(213)
    keep_idxs = {0, 100} | set(range(120, 210, 10)) | set(range(203, 213))
    correct = set(range(213)) - keep_idxs
    assert delete_idxs == list(correct)
