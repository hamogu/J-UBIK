import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

if False:
    index_and_keys = {0: 'F2100W',
                      1: 'F1800W',
                      2: 'F1500W',
                      3: 'F1280W',
                      4: 'F1000W',
                      5: 'F770W',
                      6: 'F560W',
                      7: 'F444W',
                      8: 'F356W',
                      9: 'F277W',
                      10: 'F200W',
                      11: 'F150W',
                      12: 'F115W'}
    source_light = np.load('./source_light_13channel_nocorr.npy')
    source_light = source_light[:, 100:300, 124:300]
    ylen, xlen = 3, 4
    min_source = 1e-3
    which = 'jwst'
    label = 'MJy/sr'

else:

    index_and_keys = {0: 'BAND 8'}
    source_light = np.load('./source_light_1channel_nocorr_band8.npy')
    # source_light = source_light[:, 100:300, 124:3{00]}
    ylen, xlen = 1, 1
    min_source = 1e+3
    which = 'band8'
    label = 'Jy/rad2'


norm_source = LogNorm(
    vmin=np.max((min_source, source_light.min())),
    vmax=source_light.max()
)
cmap = 'inferno'

# Create figure with space for colorbar
fig = plt.figure(figsize=(2*xlen, 2*ylen + 0.3), dpi=300)

# Create a gridspec that reserves space for the colorbar
gs = plt.GridSpec(ylen + 1, xlen, height_ratios=[0.1] + [1]*ylen)

# Create axes for all panels
axes = []
for y in range(ylen):
    for x in range(xlen):
        ax = fig.add_subplot(gs[y+1, x])
        axes.append(ax)
axes = np.array(axes)

# Create the images
ims = np.zeros_like(axes)
axes = axes.flatten()
ims = ims.flatten()

# Plot each field
for ii, (_, fld) in enumerate(zip(ims, source_light)):
    # Add title inside the panel at bottom left with transparent background
    axes[ii].text(0.05, 0.05, f'{index_and_keys[ii]}',
                  transform=axes[ii].transAxes,
                  color='white',
                  fontsize=10,
                  verticalalignment='bottom',
                  bbox=dict(facecolor='none', alpha=0))

    # Remove axis labels and ticks
    axes[ii].set_xticks([])
    axes[ii].set_yticks([])

    # Create image
    ims[ii] = axes[ii].imshow(fld, origin='lower', norm=norm_source, cmap=cmap)

# Add a single colorbar at the top
cax = fig.add_subplot(gs[0, :])
cbar = plt.colorbar(ims[0], cax=cax, orientation='horizontal', label=label)
cax.xaxis.set_label_position('top')
cax.xaxis.set_ticks_position('top')

# Adjust layout
plt.tight_layout()
fig.savefig(f'multifrequency_plot_{which}.png', dpi=300)
plt.close()
# plt.show()
