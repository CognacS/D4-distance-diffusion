import matplotlib.pyplot as plt

from collections import OrderedDict


def normalize_hist(hist_dict):
    total = sum(hist_dict.values())
    if total == 0:
        return
    hist_dict = {k: v/total for k, v in hist_dict.items()}
    return hist_dict

def values_to_hist_dict(value_list):
    hist_dict = {}
    
    for v in value_list:
        if v in hist_dict:
            hist_dict[v] += 1
        else:
            hist_dict[v] = 1
    return hist_dict


def plot_histogram_from_hist(hist_dict, normalize=True):

    if normalize:
        hist_dict = normalize_hist(hist_dict)

    plt.bar(
        x = hist_dict.keys(),
        height = hist_dict.values(),
        align='center',
        alpha=0.5
    )

    plt.show()

    return hist_dict


def plot_histogram_from_values(value_list, normalize=True):

    hist_dict = values_to_hist_dict(value_list)
    hist_dict = plot_histogram_from_hist(hist_dict, normalize=normalize)

    return hist_dict


from copy import deepcopy


def plot_histograms(*hists, names=None, normalize=True):

    # make a copy of all histograms
    hists = [deepcopy(h) for h in hists]

    # first of all, preprocess histograms
    for i in range(len(hists)):
        h = hists[i]
        if isinstance(h, list):
            h = values_to_hist_dict(h)
        if normalize:
            h = normalize_hist(h)
        hists[i] = h


    # get all keys from all histograms
    all_keys = set()

    for h in hists:
        all_keys.update(h.keys())

    # sort keys in ascending order
    all_keys = list(all_keys)
    all_keys.sort()

    # add missing keys to all histograms
    for h in hists:
        for k in all_keys:
            if k not in h:
                h[k] = 0

    # align all histograms
    for i in range(len(hists)):
        h = hists[i]
        h = {k: v for k, v in sorted(h.items(), key=lambda item: item[0])}
        hists[i] = h

    # plot
    for i in range(len(hists)):
        h = hists[i]
        plt.bar(
            x = h.keys(),
            height = h.values(),
            align='center',
            alpha=0.5,
            label=names[i] if names is not None else None
        )

    plt.legend()

    plt.show()

    return hists