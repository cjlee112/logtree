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
    
