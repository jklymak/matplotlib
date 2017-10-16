from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six
import warnings

import numpy as np

from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredOffsetbox, DrawingArea
from matplotlib.patches import Rectangle
import matplotlib.gridspec as gridspec
from matplotlib import ticker, rcParams


def example_plot(ax, fontsize=12, nodec=False):
    ax.plot([1, 2])
    ax.locator_params(nbins=3)
    if not nodec:
        ax.set_xlabel('x-label', fontsize=fontsize)
        ax.set_ylabel('y-label', fontsize=fontsize)
        ax.set_title('Title', fontsize=fontsize)
    else:
        ax.set_xticklabels('')
        ax.set_yticklabels('')


def example_pcolor(ax, fontsize=12):
    dx, dy = 0.6, 0.6
    y, x = np.mgrid[slice(-3, 3 + dy, dy),
                    slice(-3, 3 + dx, dx)]
    z = (1 - x / 2. + x ** 5 + y ** 3) * np.exp(-x ** 2 - y ** 2)
    pcm = ax.pcolormesh(x, y, z, cmap='RdBu_r', vmin=-1., vmax=1.,
                        rasterized=True)
    # ax.locator_params(nbins=3)
    ax.set_xlabel('x-label', fontsize=fontsize)
    ax.set_ylabel('y-label', fontsize=fontsize)
    ax.set_title('Title', fontsize=fontsize)
    return pcm


@image_comparison(baseline_images=['constrained_layout1'])
def test_constrained_layout1():
    'Test constrained_layout for a single subplot'
    fig = plt.figure(constrained_layout=True)
    ax = fig.add_subplot(111)
    example_plot(ax, fontsize=24)


@image_comparison(baseline_images=['constrained_layout2'])
def test_constrained_layout2():
    'Test constrained_layout for 2x2 subplots'
    fig, axs = plt.subplots(2, 2, constrained_layout=True)
    for ax in axs.flatten():
        example_plot(ax, fontsize=24)


@image_comparison(baseline_images=['constrained_layout3'])
def test_constrained_layout3():
    'Test constrained_layout for colorbars with subplots'
    fig, axs = plt.subplots(2, 2, constrained_layout=True)
    for nn, ax in enumerate(axs.flatten()):
        pcm = example_pcolor(ax, fontsize=24)
        if nn == 3:
            pad = 0.08
        else:
            pad = 0.02  # default
        fig.colorbar(pcm, ax=ax, pad=pad)


@image_comparison(baseline_images=['constrained_layout4'])
def test_constrained_layout4():
    'Test constrained_layout for a single colorbar with subplots'
    fig, axs = plt.subplots(2, 2, constrained_layout=True)
    for ax in axs.flatten():
        pcm = example_pcolor(ax, fontsize=24)
    fig.colorbar(pcm, ax=axs, pad=0.01, shrink=0.6)


@image_comparison(baseline_images=['constrained_layout5'],
                  tol=5.e-2)
def test_constrained_layout5():
    '''
    Test constrained_layout for a single colorbar with subplots,
    colorbar bottom
    '''
    fig, axs = plt.subplots(2, 2, constrained_layout=True)
    for ax in axs.flatten():
        pcm = example_pcolor(ax, fontsize=24)
    fig.colorbar(pcm, ax=axs,
                 use_gridspec=False, pad=0.01, shrink=0.6,
                 location='bottom')


@image_comparison(baseline_images=['constrained_layout6'])
def test_constrained_layout6():
    'Test constrained_layout for nested gridspecs'
    fig = plt.figure(constrained_layout=True)
    gs = gridspec.GridSpec(1, 2, fig=fig)
    gsl = gridspec.GridSpecFromSubplotSpec(2, 2, gs[0])
    gsr = gridspec.GridSpecFromSubplotSpec(1, 2, gs[1])
    axsl = []
    for gs in gsl:
        ax = fig.add_subplot(gs)
        axsl += [ax]
        example_plot(ax, fontsize=12)
    ax.set_xlabel('x-label\nMultiLine')
    axsr = []
    for gs in gsr:
        ax = fig.add_subplot(gs)
        axsr += [ax]
        pcm = example_pcolor(ax, fontsize=12)

    fig.colorbar(pcm, ax=axsr,
                 pad=0.01, shrink=0.99, location='bottom',
                 ticks=ticker.MaxNLocator(nbins=5))


@image_comparison(baseline_images=['constrained_layout8'])
def test_constrained_layout8():
    'Test for gridspecs that are not completely full'
    fig = plt.figure(figsize=(7, 4), constrained_layout=True)
    gs = gridspec.GridSpec(3, 5, fig=fig)
    axs = []
    j = 1
    for i in [0, 1]:
        ax = fig.add_subplot(gs[j, i])
        axs += [ax]
        pcm = example_pcolor(ax, fontsize=10)
        if i > 0:
            ax.set_ylabel('')
        if j < 1:
            ax.set_xlabel('')
        ax.set_title('')
    j = 0
    for i in [2, 4]:
        ax = fig.add_subplot(gs[j, i])
        axs += [ax]
        pcm = example_pcolor(ax, fontsize=10)
        if i > 0:
            ax.set_ylabel('')
        if j < 1:
            ax.set_xlabel('')
        ax.set_title('')
    ax = fig.add_subplot(gs[2, :])
    axs += [ax]
    pcm = example_pcolor(ax, fontsize=10)

    fig.colorbar(pcm, ax=axs, pad=0.01, shrink=0.6)


@image_comparison(baseline_images=['constrained_layout7'])
def test_constrained_layout7():
    'Test for proper warning if fig not set in GridSpec'
    fig = plt.figure(tight_layout=True)
    gs = gridspec.GridSpec(1, 2)
    gsl = gridspec.GridSpecFromSubplotSpec(2, 2, gs[0])
    gsr = gridspec.GridSpecFromSubplotSpec(1, 2, gs[1])
    axsl = []
    for gs in gsl:
        ax = fig.add_subplot(gs)
        axsl += [ax]
        example_plot(ax, fontsize=12)
    ax.set_xlabel('x-label\nMultiLine')
    axsr = []
    for gs in gsr:
        ax = fig.add_subplot(gs)
        axsr += [ax]
        pcm = example_pcolor(ax, fontsize=12)

    fig.colorbar(pcm, ax=axsr, pad=0.01,
                 shrink=0.99, location='bottom',
                 ticks=ticker.MaxNLocator(nbins=5))


@image_comparison(baseline_images=['constrained_layout8'])
def test_constrained_layout8():
    'Test for gridspecs that are not completely full'
    fig = plt.figure(figsize=(10, 5), constrained_layout=True)
    gs = gridspec.GridSpec(3, 5, fig=fig)
    axs = []
    j = 1
    for i in [0, 4]:
        ax = fig.add_subplot(gs[j, i])
        axs += [ax]
        pcm = example_pcolor(ax, fontsize=9)
        if i > 0:
            ax.set_ylabel('')
        if j < 1:
            ax.set_xlabel('')
        ax.set_title('')
    j = 0
    for i in [1]:
        ax = fig.add_subplot(gs[j, i])
        axs += [ax]
        pcm = example_pcolor(ax, fontsize=9)
        if i > 0:
            ax.set_ylabel('')
        if j < 1:
            ax.set_xlabel('')
        ax.set_title('')
    ax = fig.add_subplot(gs[2, :])
    axs += [ax]
    pcm = example_pcolor(ax, fontsize=9)

    fig.colorbar(pcm, ax=axs, pad=0.01, shrink=0.6)


@image_comparison(baseline_images=['constrained_layout9'])
def test_constrained_layout9():
    'Test for handling suptitle and for sharex and sharey'
    fig, axs = plt.subplots(2, 2, constrained_layout=True,
                            sharex=False, sharey=False)
    # ax = fig.add_subplot(111)
    for ax in axs.flatten():
        pcm = example_pcolor(ax, fontsize=24)
        ax.set_xlabel('')
        ax.set_ylabel('')
    ax.set_aspect(2.)
    fig.colorbar(pcm, ax=axs, pad=0.01, shrink=0.6)
    fig.suptitle('Test Suptitle', fontsize=28)


@image_comparison(baseline_images=['constrained_layout10'])
def test_constrained_layout10():
    'Test for handling legend outside axis'
    fig, axs = plt.subplots(2, 2, constrained_layout=True)
    for ax in axs.flatten():
        ax.plot(np.arange(12), label='This is a label')
    ax.legend(loc='center left', bbox_to_anchor=(0.8, 0.5))


@image_comparison(baseline_images=['constrained_layout11'],
        extensions=['png', 'pdf'])
def test_constrained_layout11():
    'Test for multiple nested gridspecs '
    fig = plt.figure(constrained_layout=True, figsize=(10, 3))
    gs0 = gridspec.GridSpec(1, 2, fig=fig)
    gsl = gridspec.GridSpecFromSubplotSpec(1, 2, gs0[0])
    gsl0 = gridspec.GridSpecFromSubplotSpec(2, 2, gsl[1])
    ax = fig.add_subplot(gs0[1])
    example_plot(ax, fontsize=9)
    axs = []
    for gs in gsl0:
        ax = fig.add_subplot(gs)
        axs += [ax]
        pcm = example_pcolor(ax, fontsize=9)
    fig.colorbar(pcm, ax=axs, shrink=0.6, aspect=70.)
    ax = fig.add_subplot(gsl[0])
    example_plot(ax, fontsize=9)


@image_comparison(baseline_images=['constrained_layout11rat'],
        extensions=['png', 'pdf'])
def test_constrained_layout11rat():
    'Test for multiple nested gridspecs with width_ratios'
    fig = plt.figure(constrained_layout=True, figsize=(10, 3))
    gs0 = gridspec.GridSpec(1, 2, fig=fig, width_ratios=[6., 1.])
    gsl = gridspec.GridSpecFromSubplotSpec(1, 2, gs0[0])
    gsl0 = gridspec.GridSpecFromSubplotSpec(2, 2, gsl[1],
            height_ratios=[2., 1.])
    ax = fig.add_subplot(gs0[1])
    example_plot(ax, fontsize=9)
    axs = []
    for gs in gsl0:
        ax = fig.add_subplot(gs)
        axs += [ax]
        pcm = example_pcolor(ax, fontsize=9)
    fig.colorbar(pcm, ax=axs, shrink=0.6, aspect=70.)
    ax = fig.add_subplot(gsl[0])
    example_plot(ax, fontsize=9)


@image_comparison(baseline_images=['constrained_layout12'],
        extensions=['png', 'pdf'])
def test_constrained_layout12():
    'Test that very unbalanced labeling still works.'
    fig = plt.figure(constrained_layout=True)

    gs0 = gridspec.GridSpec(6, 2, fig=fig)

    ax1 = fig.add_subplot(gs0[:3, 1])
    ax2 = fig.add_subplot(gs0[3:, 1])

    example_plot(ax1, fontsize=24)
    example_plot(ax2, fontsize=24)

    ax = fig.add_subplot(gs0[0:2, 0])
    example_plot(ax, nodec=True)
    ax = fig.add_subplot(gs0[2:4, 0])
    example_plot(ax, nodec=True)
    ax = fig.add_subplot(gs0[4:, 0])
    example_plot(ax, nodec=True)
    ax.set_xlabel('x-label')


@image_comparison(baseline_images=['constrained_layout13'])
def test_constrained_layout13():
    'Test that padding works.'
    fig, axs = plt.subplots(2, 2, constrained_layout=True)
    for ax in axs.flatten():
        pcm = example_pcolor(ax, fontsize=12)
        fig.colorbar(pcm, ax=ax, shrink=0.6, aspect=20., pad=0.02)
    fig.set_constrained_layout_pads(w_pad=24./72., h_pad=24./72.)


@image_comparison(baseline_images=['constrained_layout14'])
def test_constrained_layout14():
    'Test that padding works.'
    fig, axs = plt.subplots(2, 2, constrained_layout=True)
    for ax in axs.flatten():
        pcm = example_pcolor(ax, fontsize=12)
        fig.colorbar(pcm, ax=ax, shrink=0.6, aspect=20., pad=0.02)
    fig.set_constrained_layout_pads(
            w_pad=3./72., h_pad=3./72.,
            hspace=0.2, wspace=0.2)


@image_comparison(baseline_images=['constrained_layout15'])
def test_constrained_layout15():
    'Test that rcparams work.'
    rcParams['figure.constrainedlayout.do'] = True
    fig, axs = plt.subplots(2, 2)
    for ax in axs.flatten():
        example_plot(ax, fontsize=12)


@image_comparison(baseline_images=['constrained_layout16'])
def test_constrained_layout16():
    'Test ax.set_position.'
    fig, ax = plt.subplots(constrained_layout=True)
    example_plot(ax, fontsize=12)
    ax2 = fig.add_axes([0.2, 0.2, 0.4, 0.4])
