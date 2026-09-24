from macoder.training_common import learning_rate


def test_schedule_endpoints():
    assert learning_rate(0, 100, 10, 0.001) == 0.0001
    assert learning_rate(9, 100, 10, 0.001) == 0.001
    assert learning_rate(10, 100, 10, 0.001) == 0.001
    assert learning_rate(99, 100, 10, 0.001) == 0.0001
