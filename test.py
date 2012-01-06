from matplotlib import pyplot
import mut
import numpy

def is_neighbor(seqID, candidates):
    match = [seqID ^ x for x in candidates]
    m = min(match)
    return (match[0] ^ m) < m




class Monitor(object):
    def __init__(self, nrun=100, n=6, length=200, maxP=.01,
                 nsample=100, scoreFunc=mut.quartet_p_value2, **kwargs):
        self.p_data = []
        self.nsample = nsample
        self.scoreFunc = scoreFunc
        for i in range(nrun):
            mut.run_test(n, length=length, maxP=maxP, searchFunc=self, **kwargs)
        self.p_data.sort()

    def __call__(self, seqID, edgeGroup, dd):
        pvals = []
        for partners in mut.gen_partners(edgeGroup):
            quartet = [c.seqID for c in partners] + [seqID]
            join = mut.calc_quartet(quartet, dd)
            i = join[0][1] # find out which partner was found
            l = [seqID, quartet[i]] + mut.exclude_one(quartet[:3], i)
            p = self.scoreFunc(l, dd, self.nsample)
            self.p_data.append((p, is_neighbor(seqID, l[1:])))
            pvals.append((p, partners[i]))
        pvals.sort()
        return pvals

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

def error_fig(x=None, y=None, xmin=1e-8, xlabel='p-value',
              ylabel='FDR',
              **kwargs):
    if x is None:
        monitor = Monitor(**kwargs)
        x, y = monitor.analyze()
    pyplot.loglog(x, y)
    pyplot.xlim(xmin=xmin)
    pyplot.xlabel(xlabel)
    pyplot.ylabel(ylabel)


    
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


def analyze_neighbors(nrun=100, n=6, length=200, maxP=.01, **kwargs):
    naybs = []
    degrees = []
    for i in range(nrun):
        root, nseq = mut.run_test(n, length=length, maxP=maxP, **kwargs)
        tree = mut.build_ete_tree(root)
        naybs += count_tree_neighbors(tree).values()
        degrees += count_children(tree)
    return naybs, degrees