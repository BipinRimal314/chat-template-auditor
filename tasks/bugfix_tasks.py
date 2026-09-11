"""Original single-file bug-repair tasks with hidden tests.

This is a proxy for the card's coding-agent claims (SWE-bench Verified 46.4),
not a reproduction of them - SWE-bench needs per-repo Docker environments. Each
task is a plausible real-world bug shape written for this eval, so none of it
can appear in any model's training data.
"""

TASKS = [
    dict(
        id="insert_point",
        spec="`insert_point(sorted_xs, x)` returns the leftmost index where x can be "
             "inserted into sorted_xs while keeping it sorted. Ties go to the left.",
        buggy='''
def insert_point(sorted_xs, x):
    lo, hi = 0, len(sorted_xs)
    while lo < hi:
        mid = (lo + hi) // 2
        if sorted_xs[mid] < x:
            lo = mid
        else:
            hi = mid
    return lo
''',
        tests='''
def test_basic():
    assert insert_point([1,3,5], 0) == 0
    assert insert_point([1,3,5], 4) == 2
    assert insert_point([1,3,5], 6) == 3
def test_ties_left():
    assert insert_point([1,3,3,3,5], 3) == 1
def test_empty():
    assert insert_point([], 7) == 0
def test_no_hang():
    assert insert_point([1,2], 2) == 1
''',
    ),
    dict(
        id="tally",
        spec="`tally(words, into=None)` counts words into a dict. Calling it twice "
             "without `into` must start from an empty count each time.",
        buggy='''
def tally(words, into={}):
    for w in words:
        into[w] = into.get(w, 0) + 1
    return into
''',
        tests='''
def test_counts():
    assert tally(["a","b","a"]) == {"a":2,"b":1}
def test_no_leak_between_calls():
    tally(["a","a","a"])
    assert tally(["b"]) == {"b":1}
def test_explicit_into():
    d = {"a": 5}
    assert tally(["a"], into=d) == {"a":6}
''',
    ),
    dict(
        id="rolling_mean",
        spec="`rolling_mean(xs, k)` returns the list of means of every consecutive "
             "window of length k. If k <= 0 or k > len(xs) it returns an empty list.",
        buggy='''
def rolling_mean(xs, k):
    out = []
    for i in range(len(xs) - k):
        window = xs[i:i+k]
        out.append(sum(window) / k)
    return out
''',
        tests='''
def test_windows():
    assert rolling_mean([1,2,3,4], 2) == [1.5, 2.5, 3.5]
def test_full_window():
    assert rolling_mean([2,4], 2) == [3.0]
def test_degenerate():
    assert rolling_mean([1,2], 0) == []
    assert rolling_mean([1,2], 5) == []
''',
    ),
    dict(
        id="overlaps",
        spec="`overlaps(a, b)` takes two closed intervals (start, end) and returns "
             "True when they share at least one point.",
        buggy='''
def overlaps(a, b):
    return a[0] < b[1] and b[0] < a[1]
''',
        tests='''
def test_touching():
    assert overlaps((1,3),(3,5)) is True
def test_disjoint():
    assert overlaps((1,2),(3,4)) is False
    assert overlaps((3,4),(1,2)) is False
def test_contained():
    assert overlaps((1,10),(4,5)) is True
    assert overlaps((4,5),(1,10)) is True
def test_reversed_args_symmetric():
    assert overlaps((1,3),(2,9)) == overlaps((2,9),(1,3))
''',
    ),
    dict(
        id="rotate_grid",
        spec="`rotate_grid(grid)` returns a NEW grid rotated 90 degrees clockwise. "
             "The input grid must not be modified and rows of the result must not "
             "alias each other.",
        buggy='''
def rotate_grid(grid):
    n = len(grid)
    row = [0] * n
    out = [row] * n
    for i in range(n):
        for j in range(n):
            out[j][n-1-i] = grid[i][j]
    return out
''',
        tests='''
def test_rotation():
    assert rotate_grid([[1,2],[3,4]]) == [[3,1],[4,2]]
def test_input_untouched():
    g = [[1,2],[3,4]]
    rotate_grid(g)
    assert g == [[1,2],[3,4]]
def test_rows_not_aliased():
    out = rotate_grid([[1,2],[3,4]])
    out[0][0] = 99
    assert out[1][0] != 99
''',
    ),
    dict(
        id="percentile",
        spec="`percentile(xs, p)` returns the p-th percentile (0<=p<=100) of xs using "
             "linear interpolation between the two nearest ranks. xs need not be sorted.",
        buggy='''
def percentile(xs, p):
    xs = sorted(xs)
    k = (len(xs) - 1) * p // 100
    f = int(k)
    c = f + 1
    if c >= len(xs):
        return xs[f]
    return xs[f] + (xs[c] - xs[f]) * (k - f)
''',
        tests='''
def test_median():
    assert percentile([1,2,3,4], 50) == 2.5
def test_ends():
    assert percentile([1,2,3,4], 0) == 1
    assert percentile([1,2,3,4], 100) == 4
def test_interpolated():
    assert abs(percentile([0,10], 25) - 2.5) < 1e-9
def test_unsorted_input():
    assert percentile([4,1,3,2], 50) == 2.5
''',
    ),
    dict(
        id="has_cycle",
        spec="`has_cycle(graph)` takes a dict of node -> list of successors for a "
             "DIRECTED graph and returns True if it contains a directed cycle.",
        buggy='''
def has_cycle(graph):
    visited = set()
    def dfs(u):
        if u in visited:
            return True
        visited.add(u)
        for v in graph.get(u, []):
            if dfs(v):
                return True
        return False
    return any(dfs(n) for n in graph)
''',
        tests='''
def test_cycle():
    assert has_cycle({"a":["b"],"b":["c"],"c":["a"]}) is True
def test_self_loop():
    assert has_cycle({"a":["a"]}) is True
def test_dag_diamond_is_not_a_cycle():
    assert has_cycle({"a":["b","c"],"b":["d"],"c":["d"],"d":[]}) is False
def test_disconnected_dag():
    assert has_cycle({"a":["b"],"b":[],"c":["b"]}) is False
''',
    ),
    dict(
        id="merge_sorted",
        spec="`merge_sorted(a, b)` merges two sorted lists into one sorted list, "
             "keeping every element including duplicates across both lists.",
        buggy='''
def merge_sorted(a, b):
    i = j = 0
    out = []
    while i < len(a) and j < len(b):
        if a[i] < b[j]:
            out.append(a[i]); i += 1
        elif a[i] > b[j]:
            out.append(b[j]); j += 1
        else:
            out.append(a[i]); i += 1; j += 1
    out.extend(a[i:])
    out.extend(b[j:])
    return out
''',
        tests='''
def test_simple():
    assert merge_sorted([1,3],[2,4]) == [1,2,3,4]
def test_keeps_cross_duplicates():
    assert merge_sorted([1,2],[2,3]) == [1,2,2,3]
def test_all_equal():
    assert merge_sorted([5,5],[5]) == [5,5,5]
def test_empty():
    assert merge_sorted([], [1]) == [1]
''',
    ),
    dict(
        id="strip_tags",
        spec="`strip_tags(s)` removes every HTML tag from s and returns the remaining "
             "text. Text between tags must be preserved.",
        buggy='''
import re
def strip_tags(s):
    return re.sub(r"<.+>", "", s)
''',
        tests='''
def test_between_tags_kept():
    assert strip_tags("<b>hi</b>") == "hi"
def test_multiple():
    assert strip_tags("<p>a</p><p>b</p>") == "ab"
def test_attributes():
    assert strip_tags('<a href="x">go</a>') == "go"
def test_plain_text():
    assert strip_tags("no tags") == "no tags"
''',
    ),
    dict(
        id="flatten",
        spec="`flatten(x)` fully flattens arbitrarily nested lists/tuples into a flat "
             "list of scalars. Strings are scalars and must never be split into chars.",
        buggy='''
def flatten(x):
    out = []
    try:
        for item in x:
            out.extend(flatten(item))
    except TypeError:
        out.append(x)
    return out
''',
        tests='''
def test_nested():
    assert flatten([1,[2,[3,[4]]]]) == [1,2,3,4]
def test_strings_are_scalars():
    assert flatten(["ab",["cd"]]) == ["ab","cd"]
def test_tuples():
    assert flatten([1,(2,3)]) == [1,2,3]
def test_empty():
    assert flatten([]) == []
''',
    ),
    dict(
        id="drop_expired",
        spec="`drop_expired(cache, now)` removes every entry of dict `cache` whose "
             "value (an expiry timestamp) is <= now, mutating and returning cache.",
        buggy='''
def drop_expired(cache, now):
    for k in cache:
        if cache[k] <= now:
            del cache[k]
    return cache
''',
        tests='''
def test_drops():
    assert drop_expired({"a":1,"b":10}, 5) == {"b":10}
def test_drops_many():
    c = {"a":1,"b":2,"c":3,"d":99}
    assert drop_expired(c, 50) == {"d":99}
def test_mutates_in_place():
    c = {"a":1}
    assert drop_expired(c, 5) is c
def test_boundary_is_expired():
    assert drop_expired({"a":5}, 5) == {}
''',
    ),
    dict(
        id="lru",
        spec="`LRU(cap)` with `.get(k)` (returns None when missing) and `.put(k,v)`. "
             "Evicts the least recently used entry when over capacity. A get counts "
             "as a use, and re-putting an existing key counts as a use.",
        buggy='''
class LRU:
    def __init__(self, cap):
        self.cap = cap
        self.d = {}
    def get(self, k):
        return self.d.get(k)
    def put(self, k, v):
        self.d[k] = v
        if len(self.d) > self.cap:
            self.d.pop(next(iter(self.d)))
''',
        tests='''
def test_evicts_lru():
    c = LRU(2); c.put("a",1); c.put("b",2); c.put("c",3)
    assert c.get("a") is None and c.get("b") == 2 and c.get("c") == 3
def test_get_counts_as_use():
    c = LRU(2); c.put("a",1); c.put("b",2)
    c.get("a")
    c.put("c",3)
    assert c.get("a") == 1 and c.get("b") is None
def test_reput_counts_as_use():
    c = LRU(2); c.put("a",1); c.put("b",2)
    c.put("a",9)
    c.put("c",3)
    assert c.get("a") == 9 and c.get("b") is None
def test_missing():
    assert LRU(1).get("zz") is None
''',
    ),
]
