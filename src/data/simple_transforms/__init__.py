from torch.utils.data import Dataset

# define decorator to make batched
def batched(fn):
    def wrapper(elem, *args, **kwargs):
        out = []
        is_single = not isinstance(elem, (list, tuple, Dataset))
        if is_single:
            elems = [elem]
        else:
            elems = elem
        for e in elems:
            out.append(fn(e, *args, **kwargs))
        
        return out[0] if is_single else out

    return wrapper