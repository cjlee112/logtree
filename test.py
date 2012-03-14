from matplotlib import pyplot
import mut
import numpy

def is_neighbor(seqID, candidates):
    match = [seqID ^ x for x in candidates]
    m = min(match)
    return (match[0] ^ m) < m


def monitor_run(n, length, maxP, nsample, scoreFunc, **kwargs):
    mr = MonitorRun(nsample, scoreFunc)
    mut.run_test(n, length=length, maxP=maxP, searchFunc=mr, **kwargs)
    return mr.p_data

class MonitorRun(object):
    def __init__(self, nsample, scoreFunc):
        self.p_data = []
        self.nsample = nsample
        self.scoreFunc = scoreFunc
    def __call__(self, seqID, edgeGroup, dd):
        pvals = []
        for partners in mut.gen_partners(edgeGroup):
            quartet = [c.seqID for c in partners] + [seqID]
            join = mut.calc_quartet(quartet, dd)
            i = join[0][1] # find out which partner was found
            l = [seqID, quartet[i]] + mut.exclude_one(quartet[:3], i)
            if self.nsample:
                p = self.scoreFunc(l, dd, self.nsample)
            else:
                p = self.scoreFunc(l, dd)
            self.p_data.append((p, is_neighbor(seqID, l[1:])))
            pvals.append((p, partners[i]))
        pvals.sort()
        return pvals


class Monitor(object):
    def __init__(self, nrun=500, n=6, length=200, maxP=.01,
                 nsample=100, scoreFunc=mut.quartet_p_value2,
                 mapFunc=map, **kwargs):
        self.p_data = []
        for l in mapFunc(CallWrapper(monitor_run, n=n, length=length,
                                     maxP=maxP, nsample=nsample,
                                     scoreFunc=scoreFunc, **kwargs),
                         range(nrun)):
            self.p_data += l
        self.p_data.sort()

    def analyze(self):
        m = 0
        x, y = [], []
        for n,t in enumerate(self.p_data):
            if not t[1]:
                m += 1
            y.append(m / (n + 1.))
            x.append(t[0])
        return x, y

    def roc(self):
        a = numpy.array([t[1] for t in self.p_data], dtype=int).cumsum()
        positives = a[-1]
        negatives = len(self.p_data) - positives
        tpr = a / float(positives)
        b = numpy.arange(len(self.p_data)) + 1 - a
        fpr = b / float(negatives)
        return fpr, tpr

class MonitorAll(MonitorRun):
    '''use this for FDR / ROC analysis on ALL possible candidates
    (i.e. all three partition candidates are scored, rather than
    just the single candidate predicted by calc_quartet()).'''
    def __call__(self, seqID, edgeGroup, dd):
        pvals = []
        for partners in mut.gen_partners(edgeGroup):
            quartet = [c.seqID for c in partners] + [seqID]
            join = mut.calc_quartet(quartet, dd)
            i = join[0][1] # find out which partner was found
            for j in range(3): # score all 3 candidates
                l = [seqID, quartet[j]] + mut.exclude_one(quartet[:3], j)
                if self.nsample:
                    p = self.scoreFunc(l, dd, self.nsample)
                else:
                    p = self.scoreFunc(l, dd)
                self.p_data.append((p, is_neighbor(seqID, l[1:])))
                if i == j:
                    pvals.append((p, partners[i]))
        pvals.sort()
        return pvals



    
def count_neighbors(root):
    tree = mut.build_ete_tree(root)
    return count_tree_neighbors(tree)

def count_tree_neighbors(tree):
    if tree.is_leaf():
        seqID = int(tree.name)
        return {seqID | 1:0}
    d = {}
    for c in tree.get_children():
        for k,v in count_tree_neighbors(c).items():
            try:
                d[k] = abs(d[k] + v) # found neighbor, so store vertex count
            except KeyError:
                if v > 0: # already found, so just copy
                    d[k] = v
                else: # not yet found neighbor, so count vertex
                    d[k] = v - 1
    return d

def count_children(tree):
    l = [len(c.get_children()) for c in tree.iter_descendants()
         if not c.is_leaf()]
    l.append(len(tree.get_children()) - 1)
    return l


class CallWrapper(object):
    def __init__(self, func, valName=None, *args, **kwargs):
        self.func = func
        self.args = args
        self.kwargs= kwargs
        self.valName = valName
    def __call__(self, val):
        if self.valName:
            d = {self.valName:val}
            d.update(self.kwargs)
        else:
            d = self.kwargs
        return self.func(*self.args, **d)

def calc_neighb_dist(n, **kwargs):
    root, nseq = mut.run_test(n, **kwargs)
    tree = mut.build_ete_tree(root)
    return count_tree_neighbors(tree).values()

def analyze_neighbors(nrun=100, n=6, length=200, maxP=.01, mapFunc=map,
                      **kwargs):
    naybs = []
    for l in mapFunc(CallWrapper(calc_neighb_dist, n=n, length=length,
                                 maxP=maxP, **kwargs), range(nrun)):
        naybs += l
    return naybs
