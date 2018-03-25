import numpy as np
import scipy.special as ssp
import matplotlib.pyplot as plt
import pandas as pd

def calcSignificance(Data, Bkg):
    zvals = []
    chi2 = 0
    for i, nD in enumerate(Data):
        nB = Bkg[i]
        if nD != 0:
            if nB > nD:
                pval = 1.-ssp.gammainc(nD+1.,nB)
            else:
                pval = ssp.gammainc(nD,nB)
            prob = 1-2*pval
            if prob > -1 and prob < 1:
                zval = np.sqrt(2.)*ssp.erfinv(prob)
            else:
                zval = np.inf
               
            if zval > 100: zval = 20
            if zval < 0: zval = 0
            if (nD < nB): zval = -zval
        else: zval = 0
            
        zvals.append(zval)
        chi2 += ((nD - nB) ** 2 / abs(nB))
    return zvals, chi2

def plotFuncVsData(xval, xerr, bkg, yval, yerr, sig, color='C0', label="Function", savefig=False):
    f, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(6,6), gridspec_kw = {'height_ratios':[3, 1]})
    ax1.set_xscale("log", nonposx='clip')
    ax1.set_yscale("log", nonposy='clip')
    ax1.errorbar(xval, yval, yerr=yerr, fmt='o', c='k', ms=2, lw=.5, capsize=1, label="ATLAS data", zorder=1)
    ax1.plot(xval, bkg, color=color, label=label, zorder=2)
    ax1.legend(frameon=False)
    ax1.set_ylabel("Events per bin")

    ax2.bar(xval, sig, width=xerr, color=color)
    ax2.axhline(0, linestyle='-', color='black', lw=.5)
    ax2.set_ybound(-4, 4)
    ax2.set_xticks([1000, 2000])
    minor_ticks = np.arange(200, 2100, 100)
    tick_labels = [''] * len(minor_ticks)
    tick_labels[0:4] = [200, 300, '', 500]
    tick_labels[8] = 1000
    tick_labels[18] = 2000
    ax2.set_xticks(minor_ticks, minor=True)
    ax2.set_xticklabels(tick_labels, minor=True)
    ax2.set_xticklabels([])
    ax2.set_xlabel(r"$m_{ll}$ [GeV]")
    ax2.set_ylabel("Significance")

    f.subplots_adjust(hspace=0)
    if savefig:
        if savefig is True: savefig = 'fig.pdf'
        f.savefig(savefig, bbox_inches='tight')
    return f

def _remove_overflow(hist, err):
    hist = hist[1:-1]
    if err is not None:
        if len(err) == len(hist) + 2: # +2 since we removed the overflow
            err = err[1:-1]
        elif len(err) == 2: # For asymmetric errors
            err = [err[0][1:-1], err[1][1:-1]]
        #TODO: What if len(hist) == 2 ?
        else:
            raise ValueError('err has wrong shape')
    return hist, err

def plot_hist(hist, bins, err=None, ax=None, has_overflow=True, no_plot=False, **kwds):
    ax = plt.gca() if ax is None and not no_plot else ax
    if has_overflow:
        hist, err = _remove_overflow(hist, err)

    left, right = bins[:-1], bins[1:]
    midpoints = (left + right) / 2.0
    x = np.array([left, right]).T.flatten()
    y = np.array([hist, hist]).T.flatten()

    if not no_plot:
        line, = ax.plot(x, y, **kwds)
        if err is not None:
            if 'color' in kwds:
                del kwds['color']
            ax.errorbar(midpoints, hist, yerr=err, fmt='none',
                        color=line.get_color(), **kwds)
        ax.set_xlim(bins[0], bins[-1])
    return x, y, err

def smear(s, smeared_bins, truth_bins, truth_err, lumi=36.1, nsmear=5):
    smeared_counts=np.zeros(len(smeared_bins))
    s=s*(lumi/36.1)
    bkg=bkg5param*(lumi/36.1)
    for n in range(nsmear):
        samp=np.concatenate((bkg, s))
        for xi, x in enumerate(smeared_bins):
            gaus = truth_err*ss.norm.pdf(x, truth_bins, 2*truth_err)
            gaus = gaus/np.sum(gaus)
            smeared_counts[xi]=np.sum(samp*gaus) 
        smeared_counts=0.97*smeared_counts
        s=np.concatenate((smeared_counts, np.zeros(len(xvalO_ext)-len(smeared_counts))))
    noisy_smeared_counts=np.random.poisson(smeared_counts)
    return smeared_counts, noisy_smeared_counts

def generate_double_peak(size, smear):
    peak1 = np.random.normal(0.3, 0.1, int(size/2))
    peak2 = np.random.normal(0.7, 0.1, int(size/2))
    truth = np.random.permutation(
        np.concatenate([peak1, peak2]))
    smeared = np.random.normal(truth, smear)

    return pd.DataFrame({'truth': truth, 'reco': smeared})