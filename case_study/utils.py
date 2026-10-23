import matplotlib.pyplot as plt
import pickle

COLOURS = plt.rcParams['axes.prop_cycle'].by_key()['color']
font_size = 14
ticks_fontsize = font_size - 2.5
legend_font_size = font_size - 2
title_font_size = font_size + 1
settings = {
    'font.size': font_size,
    'font.family': "serif", 
    "mathtext.fontset": "dejavuserif",  #cm
    'figure.titlesize': title_font_size,
    'figure.autolayout': True,
    'legend.fontsize': legend_font_size,
    'axes.titlesize': font_size,
    'axes.labelsize': font_size - 1,
    'ytick.labelsize': ticks_fontsize,
    'xtick.labelsize': ticks_fontsize,
    'hatch.linewidth': 0.8,
    'xtick.minor.pad': 1,
    'axes.labelpad': 3,
    'axes.grid': True,
    'grid.linestyle': '--',
    'grid.linewidth': 0.4,
    'grid.alpha': 0.8,
    'axes.axisbelow': True,
    'legend.framealpha': 1,
    'legend.edgecolor': 'black',
    'legend.fancybox': False,
    'legend.handletextpad': 0.2,
    'legend.columnspacing': 0.5,
    'figure.dpi': 800,
    'figure.max_open_warning': 0,
    'legend.facecolor': 'white',
    'lines.linewidth': 1,
    'errorbar.capsize': 3,  
    'lines.markeredgewidth': 0.5, 
    'lines.markersize': 3,
    'axes.formatter.limits':  [-3, 3],
}
plt.rcParams.update(settings)

def load_pkl(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj
    