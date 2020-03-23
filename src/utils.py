def format_metrics(metrics, prefix=''):
    return ", ".join([
            "{}: {:.4f}".format(prefix+metric_name, metric_values)
            for metric_name, metric_values in metrics.items()
        ])

