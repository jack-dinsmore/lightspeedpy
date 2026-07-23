import numpy as np
import os
import matplotlib.pyplot as plt
from ..dataset import DataSet
from ..pixel_properties import PixelProperties, ADU_PER_ELECTRON

def get_dataset(args):
    if os.path.exists(args.output) and not args.clobber:
        raise Exception(f"Cannot save to {args.output}: file already exists and clobber is False")
    
    min_index = None if args.min_index is None else int(args.min_index)
    max_index = None if args.max_index is None else int(args.max_index)
    data_set = DataSet.from_first(args.input, min_index=min_index, max_index=max_index, cut_cr=False, bar_color='green')

    print("Loaded files")
    data_set.display_filenames()
        
    return data_set

def stack_bias(args):
    data_set = get_dataset(args)
    pp = PixelProperties.from_bias(data_set, data_set, args.map_noise)
    pp.save(args.output, args.clobber)

    if args.dbg_noise and args.map_noise:
        n_display = 100
        edges = np.arange(-2, 2, 1/ADU_PER_ELECTRON)
        centers = (edges[1:] + edges[:-1]) / 2
        n_pixels = np.prod(data_set.image_shape)
        counts = np.zeros((len(edges)+1, n_display), int)
        arange = np.arange(n_display)

        # Make mask of randomly selected pixels
        pixel_indices = np.random.choice(np.arange(n_pixels), n_display, replace=False)
        mask = np.zeros(n_pixels, bool)
        mask[pixel_indices] = True
        mask = mask.reshape(*data_set.image_shape)
        
        # Get histograms
        for frame in data_set.iterator(max_frames=10_000):
            digits = np.digitize(frame.image[mask], edges)
            counts[digits, arange] += 1
        counts = counts[1:-1,:]

        # Get model curves
        probabilities = np.array([pp.get_prob(np.ones(n_display) * c, 0, mask) for c in edges])

        fig, axs = plt.subplots(ncols=10, nrows=10, sharex=True, sharey=True, figsize=(18,12))
        axs = axs.reshape(-1)
        top = 0
        for (ax, hist, probs, params) in zip(axs, counts.transpose(), probabilities.transpose(), pp.params[mask]):
            ax.step(centers, hist, where="mid", color='k')
            ax.plot(edges, probs * np.sum(hist) / np.sum(probs), color='C1')
            ax.text(0, 1, '\n'.join([f"{p:.2f}" for p in params]), ha="left", va="top", transform=ax.transAxes, size=6)
            top = max(top, np.max(hist))
        ax.set_yscale("log")
        ax.set_ylim(0.9, top * 2)
        fig.subplots_adjust(hspace=0, wspace=0)
        fig.savefig("dbg-noise.png")