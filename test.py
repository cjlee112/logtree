from matplotlib import pyplot
import mut

def is_neighbor(seqID, candidates):
    match = [seqID ^ x for x in candidates]
    m = min(match)
    return (match[0] ^ m) < m




class Monitor(object):
    def __init__(self, n=10):
        self.p_data = []
        self.n = n
    def __call__(self, seqID, edgeGroup, dd):
        pvals = []
        for partners in mut.gen_partners(edgeGroup):
            quartet = [c.seqID for c in partners] + [seqID]
            join = mut.calc_quartet(quartet, dd)
            i = join[0][1] # find out which partner was found
            l = [seqID, quartet[i]] + mut.exclude_one(quartet[:3], i)
            p = mut.quartet_p_value2(l, dd, self.n)
            self.p_data.append((p, is_neighbor(seqID, l[1:])))
            pvals.append((p, partners[i]))
        pvals.sort()
        return pvals

    def analyze(self):
        self.p_data.sort()
        m = 0
        x, y = [], []
        for n,t in enumerate(self.p_data):
            if not t[1]:
                m += 1
            y.append(m / (n + 1.))
            x.append(t[0])
        return x, y

def analyze_errors(nrun=10, n=6, length=200, maxP=.01, nsample=10, **kwargs):
    monitor = Monitor(nsample)
    for i in range(nrun):
        mut.run_test(n, length=length, maxP=maxP, searchFunc=monitor, **kwargs)
    return monitor.analyze()

def error_fig(x=None, y=None, xlabel='p-value',
              ylabel='cumulative error probability',
              **kwargs):
    if x is None:
        x, y = analyze_errors(**kwargs)
    pyplot.semilogx(x, y)
    pyplot.xlabel(xlabel)
    pyplot.ylabel(ylabel)
    
