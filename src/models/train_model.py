import itertools

def generate_all_combinations(d):
    """All permutations of dict elements.
    
    Source:
        https://stackoverflow.com/questions/38721847/how-to-generate-all-combination-from-values-in-dict-of-lists-in-python
    """
    keys, values = zip(*d.items())
    return [dict(zip(keys, v)) for v in itertools.product(*values)]