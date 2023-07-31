from autorad.data import ImageDataset as OrigImageDataset
import logging
import matplotlib.pyplot as plt

from autorad.visualization import matplotlib_utils, plot_volumes

log = logging.getLogger(__name__)


class ImageDataset(OrigImageDataset):
    def plot_examples(self, n: int = 1, window="soft tissues", label=1):
        if n > len(self.image_paths):
            n = len(self.image_paths)
            log.info(
                f"Not enough cases. Plotting all the cases instead (n={n})"
            )
        df_to_plot = self.df.sample(n)
        nrows, ncols, figsize = matplotlib_utils.get_subplots_dimensions(n)
        fig, axs = plt.subplots(nrows, ncols, figsize=figsize)
        for i in range(len(df_to_plot)):
            case = df_to_plot.iloc[i]
            ax = axs.flat[i]
            vols = plot_volumes.BaseVolumes.from_nifti(
                case[self.image_colname],
                case[self.mask_colname],
                window=window,
                label=label
            )
            image_2D, mask_2D = vols.get_slices()
            single_plot = plot_volumes.overlay_mask_contour(image_2D, mask_2D)
            ax.imshow(single_plot)
            ax.set_title(f"{case[self.ID_colname]}")
            ax.axis("off")
            # return fig
            ax.axis("off")
            # return fig
            ax.axis("off")
            # return fig
            ax.axis("off")
        # return fig
