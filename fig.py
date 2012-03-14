import test
import mut
from math import log, exp
from matplotlib import pyplot

def linear_fig(sizes, times, xlabel='number of sequences',
               ylabel='CPU time (sec)'):
    pyplot.plot(sizes, times, marker='o')
    pyplot.xlabel(xlabel)
    pyplot.ylabel(ylabel)

def loglog_fig(sizes, times, xlabel='n log(n)',
               ylabel='CPU time (sec)'):
    xdata = [x * log(x) for x in sizes]
    pyplot.loglog(xdata, times, 'bo')
    xavg = exp(sum([log(x) for x in xdata]) / len(xdata))
    yavg = exp(sum([log(x) for x in times]) / len(times))
    line = [yavg * x / xavg for x in xdata]
    pyplot.loglog(xdata, line, 'k--')
    pyplot.xlabel(xlabel)
    pyplot.ylabel(ylabel)

def plot_cubic_time(sizes, times):
    xdata = [x * log(x) for x in sizes]
    t0 = times[0] / (sizes[0] * sizes[0] * sizes[0])
    cubic = [t0 * x * x * x for x in sizes]
    pyplot.loglog(xdata, cubic, color='r')

def plot_total_pairs(sizes):
    xdata = [x * log(x) for x in sizes]
    pairs = [x * (x-1) / 2 for x in sizes]
    pyplot.loglog(xdata, pairs, color='r')

def time_data(r=range(4, 14), maxP=.01, **kwargs):
    return mut.test_range(r, maxP=maxP, **kwargs)

def time_fig(sizes, times, distances, nseqs):
    pyplot.subplot(311)
    linear_fig(sizes, times)
    pyplot.subplot(312)
    loglog_fig(sizes, times)
    plot_cubic_time(sizes, times)
    pyplot.subplot(313)
    loglog_fig(sizes, distances, ylabel='number of distances')
    plot_total_pairs(sizes)
    
def error_fig(x=None, y=None, xmin=1e-8, xlabel='p-value',
              ylabel='FDR', plotargs={},
              **kwargs):
    if x is None:
        monitor = Monitor(**kwargs)
        x, y = monitor.analyze()
    pyplot.loglog(x, y, **plotargs)
    pyplot.xlim(xmin=xmin)
    pyplot.xlabel(xlabel)
    pyplot.ylabel(ylabel)

def error_data(mapFunc=map, **kwargs):
    monitor = test.Monitor(scoreFunc=mut.quartet_p_value_gmean,
                           nsample=None, mapFunc=mapFunc, **kwargs)
    monitor2 = test.Monitor(mapFunc=mapFunc, **kwargs)
    monitor3 = test.Monitor(scoreFunc=mut.quartet_p_value2_mean,
                            mapFunc=mapFunc, **kwargs)
    return monitor, monitor2, monitor3

def roc_figure(monitor, monitor2, monitor3, xlabel='FPR',
               ylabel='TPR'):
    fpr, tpr = monitor2.roc()
    pyplot.plot(fpr, tpr)
    fpr, tpr = monitor.roc()
    pyplot.plot(fpr, tpr, color='r', linestyle=':')
    fpr, tpr = monitor3.roc()
    pyplot.plot(fpr, tpr, color='g', linestyle='-.')
    pyplot.plot((0.,1.),(0.,1.), color='k', linestyle='--')
    pyplot.xlabel(xlabel)
    pyplot.ylabel(ylabel)

def roc_all_fig(**kwargs):
    'ROC plot using MonitorAll data'
    monitorA, monitor2A, monitor3A = fig.error_data(monitorClass=test.MonitorAll, **kwargs)
    roc_figure(monitorA, monitor2A, monitor3A)

def error_fig2(monitor, monitor2, monitor3):
    pyplot.subplot(211)
    x, y = monitor.analyze()
    pyplot.loglog(x, y, color='r', linestyle=':')
    x, y = monitor3.analyze()
    pyplot.loglog(x, y, color='g', linestyle='-.')
    x, y = monitor2.analyze()
    error_fig(x, y, plotargs=dict(color='b'))
    pyplot.subplot(212)
    roc_figure(monitor, monitor2, monitor3)

def neighbor_data(r=range(200,1001, 100), **kwargs):
    l = []
    for length in r:
        naybs, degrees = test.analyze_neighbors(length=length, **kwargs)
        l.append(sum(naybs) / float(len(naybs)))
    return l

def histogram_data(keys, naybs):
    l = []
    for k in keys:
        m = []
        for vals in naybs:
            m.append(sum([i==k for i in vals]) / float(len(vals)))
        l.append(m)
    return l

def calc_mean_dist(naybs):
    return [sum(vals) / float(len(vals)) for vals in naybs]

def neighb_fig1(x, dists1, dists2, xlabel='length',
                ylabel='mean neighbor distance', xmax=1000, xmin=None):
    if xmin is None:
        xmin = x[0]
    pyplot.plot(x, dists1, marker='+', color='r', linestyle='--')
    pyplot.plot(x, dists2, marker='o', color='b')
    pyplot.xlim(xmin=xmin, xmax=xmax)
    pyplot.xlabel(xlabel)
    pyplot.ylabel(ylabel)
    
def neighb_fig2(x, histNaive, histDR, histNDR1, histNDR2, xlabel='length',
                ylabel='Fraction of Neighbors Matched', xmax=1000, xmin=None):
    if xmin is None:
        xmin = x[0]
    pyplot.plot(x, histNaive, marker='+', color='r', linestyle='--')
    pyplot.plot(x, histDR, marker='o', color='b')
    pyplot.plot(x, histNDR1, marker='^', color='g', linestyle=':')
    pyplot.plot(x, histNDR2, marker='s', color='k', linestyle='-.')
    pyplot.xlim(xmin=xmin, xmax=xmax)
    pyplot.xlabel(xlabel)
    pyplot.ylabel(ylabel)

def neighb_composite(x, dists1, dists2, histNaive, histDR, histNDR1, histNDR2):
    pyplot.subplot(211)
    neighb_fig1(x, dists1, dists2)
    pyplot.subplot(212)
    neighb_fig2(x, histNaive, histDR, histNDR1, histNDR2)
    
