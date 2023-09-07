from evaluation.utils import Metrics


def test_metrics_handles_new_metric_names():
    # Given
    metrics: Metrics = Metrics()

    # When
    metrics.add('accuracy', 0.5)

    # Then
    assert list(metrics.keys()) == ['accuracy']
    assert list(metrics.values()) == [[0.5]]

def test_metrics_add_many():
    # Given
    metrics: Metrics = Metrics()
    metrics.add('accuracy', 0.3)
    metrics.add('accuracy', 0.1)
    metrics.add('recall', 0.7)
    new_metrics = {
        'accuracy': [0.7, 0.9, 0.5],
        'mrr': [0.2, 0.3, 0.4],
        'recall': 0.1,
        'f1': 1
    }

    # When
    metrics.add_many(new_metrics)

    # Then
    expected_summary = {
        'accuracy': 0.5,
        'mrr': 0.3,
        'recall': 0.85,
        'f1': 1
    }


def test_metrics_summarize():
    # Given
    metrics: Metrics = Metrics()
    metrics.add('accuracy', 0)
    metrics.add('accuracy', 1)
    metrics.add('mrr', 0.7)

    # When
    summmary: dict = metrics.summarize()

    # Then
    expected_summary = {
        'accuracy': 0.5,
        'mrr': 0.7
    }
    assert summmary == expected_summary