import numpy as np

def summarize_acc(correct_by_groups, total_by_groups,
                  stdout=True, return_groups=False):
    all_correct = 0
    all_total = 0
    min_acc = 101.
    min_correct_total = [None, None]
    groups_accs = np.zeros([len(correct_by_groups),
                            len(correct_by_groups[-1])])
    if stdout:
        print('Accuracies by groups:')
    for yix, y_group in enumerate(correct_by_groups):
        for aix, a_group in enumerate(y_group):
            acc = a_group / total_by_groups[yix][aix] * 100
            groups_accs[yix][aix] = acc
            # Don't report min accuracy if there's no group datapoints
            if acc < min_acc and total_by_groups[yix][aix] > 0:
                min_acc = acc
                min_correct_total[0] = a_group
                min_correct_total[1] = total_by_groups[yix][aix]
            if stdout:
                print(
                    f'{yix}, {aix}  acc: {int(a_group):5d} / {int(total_by_groups[yix][aix]):5d} = {a_group / total_by_groups[yix][aix] * 100:>7.3f}')
            all_correct += a_group
            all_total += total_by_groups[yix][aix]
    if stdout:
        average_str = f'Average acc: {int(all_correct):5d} / {int(all_total):5d} = {100 * all_correct / all_total:>7.3f}'
        robust_str = f'Robust  acc: {int(min_correct_total[0]):5d} / {int(min_correct_total[1]):5d} = {min_acc:>7.3f}'
        print('-' * len(average_str))
        print(average_str)
        print(robust_str)
        print('-' * len(average_str))

    avg_acc = all_correct / all_total * 100

    if return_groups:
        return avg_acc, min_acc, groups_accs
    return avg_acc, min_acc


def summarize_acc_from_predictions(predictions, dataloader,
                                   args, stdout=True,
                                   return_groups=False):
    targets_t = dataloader.dataset.targets_all['target']
    targets_s = dataloader.dataset.targets_all['spurious']

    correct_by_groups = np.zeros([args.num_classes,
                                  args.num_classes])
    total_by_groups = np.zeros(correct_by_groups.shape)

    all_correct = (predictions == targets_t)
    for ix, s in enumerate(targets_s):
        y = targets_t[ix]
        correct_by_groups[int(y)][int(s)] += all_correct[ix]
        total_by_groups[int(y)][int(s)] += 1
    return summarize_acc(correct_by_groups, total_by_groups,
                         stdout=stdout, return_groups=return_groups)

def error_and_gap(predictions, dataloader, args, verbose=True):
    if args.dataset == 'celebA':
        try:
            predictions = predictions.cpu().numpy()
        except:
            pass
        avg_acc, min_acc = summarize_acc_from_predictions(
            predictions, dataloader, args, stdout=verbose
        )
    else:
        print('Not Support')
        avg_acc, min_acc = None, None
    return avg_acc, min_acc