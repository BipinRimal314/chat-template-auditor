"""Reference fixes. Used only to prove every hidden test suite is satisfiable."""
REFERENCE = {
"insert_point": '''
def insert_point(sorted_xs, x):
    lo, hi = 0, len(sorted_xs)
    while lo < hi:
        mid = (lo + hi) // 2
        if sorted_xs[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    return lo
''',
"tally": '''
def tally(words, into=None):
    if into is None:
        into = {}
    for w in words:
        into[w] = into.get(w, 0) + 1
    return into
''',
"rolling_mean": '''
def rolling_mean(xs, k):
    if k <= 0 or k > len(xs):
        return []
    return [sum(xs[i:i+k]) / k for i in range(len(xs) - k + 1)]
''',
"overlaps": '''
def overlaps(a, b):
    return a[0] <= b[1] and b[0] <= a[1]
''',
"rotate_grid": '''
def rotate_grid(grid):
    n = len(grid)
    out = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            out[j][n-1-i] = grid[i][j]
    return out
''',
"percentile": '''
def percentile(xs, p):
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100.0
    f = int(k)
    c = f + 1
    if c >= len(xs):
        return xs[f]
    return xs[f] + (xs[c] - xs[f]) * (k - f)
''',
"has_cycle": '''
def has_cycle(graph):
    WHITE, GREY, BLACK = 0, 1, 2
    color = {}
    def dfs(u):
        color[u] = GREY
        for v in graph.get(u, []):
            c = color.get(v, WHITE)
            if c == GREY:
                return True
            if c == WHITE and dfs(v):
                return True
        color[u] = BLACK
        return False
    return any(color.get(n, WHITE) == WHITE and dfs(n) for n in graph)
''',
"merge_sorted": '''
def merge_sorted(a, b):
    i = j = 0
    out = []
    while i < len(a) and j < len(b):
        if a[i] <= b[j]:
            out.append(a[i]); i += 1
        else:
            out.append(b[j]); j += 1
    out.extend(a[i:]); out.extend(b[j:])
    return out
''',
"strip_tags": '''
import re
def strip_tags(s):
    return re.sub(r"<[^>]*>", "", s)
''',
"flatten": '''
def flatten(x):
    if isinstance(x, (list, tuple)):
        out = []
        for item in x:
            out.extend(flatten(item))
        return out
    return [x]
''',
"drop_expired": '''
def drop_expired(cache, now):
    for k in [k for k, v in cache.items() if v <= now]:
        del cache[k]
    return cache
''',
"lru": '''
from collections import OrderedDict
class LRU:
    def __init__(self, cap):
        self.cap = cap
        self.d = OrderedDict()
    def get(self, k):
        if k not in self.d:
            return None
        self.d.move_to_end(k)
        return self.d[k]
    def put(self, k, v):
        self.d[k] = v
        self.d.move_to_end(k)
        if len(self.d) > self.cap:
            self.d.popitem(last=False)
''',
}
