import numpy as np
import torch

def format_metrics(metrics, prefix=''):
    return ", ".join([
        "{}: {:.4f}".format(prefix+metric_name, metric_values)
        for metric_name, metric_values in metrics.items()
    ])

def print_confusion_matrix(confusion_matrix, classes, normalize=False):
    sum_predictions = confusion_matrix.sum(0)
    sum_targets = confusion_matrix.sum(1)
    diag = confusion_matrix.diag().float()
    precisions = (diag/sum_predictions).numpy()
    recalls = (diag/sum_targets).numpy()
    max_value = torch.max(confusion_matrix).item()

    if normalize:
        confusion_matrix = confusion_matrix.float()
    else:
        confusion_matrix = confusion_matrix.long()

    #row_sums = torch.sum(confusion_matrix, 1).long()
    if normalize:
        value_format = "{:.3f}"
        value_length = 4

        normalization_value = int(confusion_matrix[:,:].sum().item())
        confusion_matrix /= normalization_value

    else:
        value_length = max(4, int(np.ceil(np.log10(max_value))))

        value_format = " {}"

    class_format = "{} |"
    longest_label = len(max(classes, key=len))
    class_length = len(class_format.format("".join([" "]*longest_label)))
    values_format = [value_format for i in range(confusion_matrix.size()[-1])]
    end_format = "{:.3f}"
    end_length = len(end_format.format(0.1)[1])

    footer_desc = ""
    footer_desc_length = len(footer_desc)
    footer_value_format = "{:.3f}"
    desc_length = max(class_length, footer_desc_length)

    for i, row in enumerate(confusion_matrix):
        class_label = classes[i]

        end_value = recalls[i]

        print(class_format.format(class_label).rjust(desc_length), end="")
        for value in row:
            print(value_format.format(value)[1:].rjust(value_length+1), end="")
        print(" |", end="")
        print(end_format.format(end_value)[1:].rjust(6))

    print()
    print(footer_desc.rjust(desc_length), end="")
    for value in precisions:
        print(footer_value_format.format(value)[1:].rjust(value_length+1), end="")
    print()
    #print(end_format.format(end_value).rjust(end_length+1))


if __name__ == '__main__':
    matrix = torch.randint(1, 3, (10, 10))
    classes = ["lol"]*5 + ["keksi"]*5

    print_confusion_matrix(matrix, classes)
    print()
    print_confusion_matrix(matrix, classes, normalize=True)
