import copy
from typing import List
from plot import compression_methods


def default_omit_methods():
    return ["RLE", "Uncompressed"]


def default_compression_method_order() -> List[str]:
    compression_methods_ordered: List[str] = copy.deepcopy(compression_methods)
    # if "Uncompressed" in compression_methods_ordered:
    #     compression_methods_ordered.remove("Uncompressed")
    for method in default_omit_methods():
        if method in compression_methods_ordered:
            compression_methods_ordered.remove(method)
    return compression_methods_ordered
