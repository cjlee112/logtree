import random
from math import exp, log
from scipy import stats
import time

def mutate(s, d, a='ATGC', h = .75):
    'mutate sequence s according the specified JC distance d'
    f = (1. - exp(-d / h))
    l = []
    for c in s:
        if random.random() < f:
            l.append(random.choice(a))
        else:
            l.append(c)
    return ''.join(l)


def random_seq(length=100, a='ATGC'):
    return ''.join([random.choice(a) for i in range(length)])

def random_tree(tree, length=100, root=None):
    '''generate sequences for the specified tree structure, which
    must be specified in the form [(d,subtree),(d,subtree)],
    where d is a JC distance, and subtree is specified in exactly
    the same tree format; a terminal leaf is specified by giving
    an empty tuple as the subtree.

    Returns a sequence tree of the form [(s,subtree),(s,subtree)],
    where s is a nucleotide string, and subtree is in this same
    format.'''
    if root is None:
        root = random_seq(length)
        return ((root, random_tree(tree, length, root)),)
    l = []
    for d, subtree in tree:
        s = mutate(root, d)
        l.append((s, random_tree(subtree, length, s)))
    return l

def print_indent(tree, indent=''):
    'prints a tree, one seq per line, indented according to its depth'
    for s, subtree in tree:
        print indent + s
        print_indent(subtree, indent + '  ')

def random_distances(d, n):
    if n == 1:
        return [(d, ()), (d, ())]
    d1 = random.random() * d * 2. / n
    d2 = random.random() * d * 2. / n
    return [(d1, random_distances(d - d1, n - 1)),
            (d2, random_distances(d - d2, n - 1))]

def fixed_distances(d, n):
    if n == 1:
        return [(d, ()), (d, ())]
    d1 = d2 = d / n
    return [(d1, fixed_distances(d - d1, n - 1)),
            (d2, fixed_distances(d - d2, n - 1))]


def root_distances(tree, d=0., l=None):
    if l is None:
        l = []
    for d1, subtree in tree:
        if subtree:
            root_distances(subtree, d + d1, l)
        else:
            l.append(d + d1)
    return l

def get_leaves(tree, l=None):
    if l is None:
        l = []
    for s, subtree in tree:
        if subtree:
            get_leaves(subtree, l)
        else:
            l.append(s)
    return l

def mutated(x, y):
    return x != y

def calc_distance(a, b, h=0.75):
    'jukes-cantor distance'
    m = sum(map(mutated, a, b))
    f = float(m) / len(a)
    return -h * log(1. - f / h), m

class DistanceDict(dict):
    'takes (i,j) pair as key, caches calculated distances'
    def __init__(self, seqs):
        dict.__init__(self)
        self.seqs = seqs
    def __getitem__(self, k):
        i,j = k
        if i > j:
            i,j = j,i
        try:
            return dict.__getitem__(self, (i,j))[0]
        except KeyError:
             d = calc_distance(self.seqs[i], self.seqs[j])
             dict.__setitem__(self, (i,j), d)
             return d[0]
    def __setitem__(self, k, d):
        i,j = k
        if i > j:
            i,j = j,i
        dict.__setitem__(self, (i,j), d)
    def get_count(self, i, j):
        'get mutation counts for this pair'
        if i > j:
            i,j = j,i
        try:
            return dict.__getitem__(self, (i,j))[1]
        except KeyError:
            self[i,j] # force distance calculation
            return dict.__getitem__(self, (i,j))[1]
        
        

qpairs = (((0,3), (1,2)),
          ((1,3), (0,2)),
          ((2,3), (0,1)))

def calc_quartet(q, dd):
    l = []
    for i,pairs in enumerate(qpairs):
        l.append((dd[q[pairs[0][0]], q[pairs[0][1]]] +
                  dd[q[pairs[1][0]], q[pairs[1][1]]], i))
    l.sort()
    return l

def quartet_p_value1(q, dd, h=0.75):
    d = (dd[q[0], q[1]] + dd[q[2], q[3]] + dd[q[0], q[2]] - dd[q[1], q[3]]) / 2.
    f = h * (1. - exp(-d / h))
    pmf = stats.binom(len(dd.seqs[q[0]]), f)
    return pmf.sf(dd.get_count(q[0], q[2]) - 1)

def quartet_p_value(q, dd, h=0.75):
    'use both non-partners mutation counts to calculate combined p-value'
    p1 = quartet_p_value1(q, dd, h)
    p2 = quartet_p_value1((q[0], q[1], q[3], q[2]), dd, h)
    p = p1 * p2
    if p == 0.:
        return p
    return p * (1. - log(p)) # Jost integral for combining 2 p-values

class PseudoEdge(object):
    def __init__(self, parentNode, terminal):
        self.parentNode = parentNode
        self.terminal = terminal
        self.origin = InnerEnd(self)
    def add_subnode(self, leaf):
        self.subnode = Node(self, leaf, maxP=self.parentNode.maxP)

class ClosestSeq(object):
    def __init__(self, seqID, d):
        self.seqID = seqID
        self.d = d
    def __cmp__(self, other):
        try:
            return cmp(self.d, other.d)
        except AttributeError:
            return cmp(id(self), id(other))

class OuterEnd(object):
    def __init__(self, seqID):
        self.seqID = seqID
    def get_closest(self):
        return self.seqID

class InnerEnd(object):
    def __init__(self, parentEdge):
        self.parentEdge = parentEdge
    def get_closest2(self):
        l = []
        for i,e in enumerate(self.parentEdge.parentNode.edges):
            if e is not self.parentEdge:
                l.append(self.parentEdge.parentNode.closest[i])
        l.sort()
        return l
    def get_closest(self):
        return self.get_closest2()[0].seqID

class Node(object):
    def __init__(self, parentEdge, leaf, leaf2=None, leaf3=None,
                 dd=None, maxP=1e-3):
        self.maxP = maxP
        if dd is None:
            dd = parentEdge.parentNode.dd
        self.dd = dd
        self.parentEdge = parentEdge
        if parentEdge is None: # model as pseudoroot
            self.edges = (PseudoEdge(self, OuterEnd(leaf)),
                          PseudoEdge(self, OuterEnd(leaf2)),
                          PseudoEdge(self, OuterEnd(leaf3)))
            self._init_edges()
            return
        self.edges = (PseudoEdge(self, InnerEnd(parentEdge)),
                      PseudoEdge(self, parentEdge.terminal),
                      PseudoEdge(self, OuterEnd(leaf)))
        self._init_edges()

    def _init_edges(self):
        seqs = [e.terminal.get_closest() for e in self.edges]
        pairD = (self.dd[seqs[1],seqs[2]],
                 self.dd[seqs[0],seqs[2]],
                 self.dd[seqs[0],seqs[1]])
        sumD = sum(pairD)
        self.closest = []
        for i,d in enumerate(pairD):
            self.closest.append(ClosestSeq(seqs[i], (sumD - 2. * d) / 2.))
        
    def add_seq(self, seqID):
        quartet = [c.seqID for c in self.closest] + [seqID]
        join = calc_quartet(quartet, self.dd)
        i = join[0][1] # find out which partner was found
        l = [seqID, quartet[i]] \
            + [quartet[j] for j in range(3) if j != i]
        p = quartet_p_value(l, self.dd)
        #print p, l
        if p > self.maxP: # ambiguous, so give up
            #print 'FAIL', join
            return 0
        try:
            subnode = self.edges[i].subnode
        except AttributeError:
            self.edges[i].add_subnode(seqID)
            #print 'SUCCESS'
            return 1
        else: # recurse to subtree
            return subnode.add_seq(seqID)

def build_tree(seqs, **kwargs):
    dd = DistanceDict(seqs)
    ids = range(1, len(seqs) - 1)
    random.shuffle(ids)
    root = Node(None, ids[0], 0, len(seqs) - 1, dd, **kwargs)
    n = 3
    for seqID in ids[1:]:
        n += root.add_seq(seqID)
    print 'tree size:', n
    return root, n

def run_test(n, d=0.3, length=10000):
    dtree = fixed_distances(d, n)
    stree = random_tree(dtree, length)
    leaves = get_leaves(stree)
    total = len(leaves)
    t = time.time()
    root, nseq = build_tree(leaves, maxP=.05)
    return total, time.time() - t, len(root.dd), nseq

def test_range(r, **kwargs):
    sizes, times, distances, nseqs = [],[],[],[]
    for n in r:
        c,t,nd,ns = run_test(n, **kwargs)
        sizes.append(c)
        times.append(t)
        distances.append(nd)
        nseqs.append(ns)
    return sizes, times, distances, nseqs

        
