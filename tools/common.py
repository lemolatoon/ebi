import copy
import json
import os
from typing import List, Literal, TypedDict

from matplotlib import pyplot as plt
import matplotlib

CompressionMethodKeys = Literal[
    "Uncompressed",
    "RLE",
    "Gorilla",
    "Chimp",
    "Chimp128",
    "ElfOnChimp",
    "Elf",
    "BUFF",
    "DeltaSprintz",
    "Zstd",
    "Gzip",
    "Snappy",
    "FFIAlp",
]

compression_methods: List[CompressionMethodKeys] = [
    "Uncompressed",
    "RLE",
    "Gorilla",
    "Chimp",
    "Chimp128",
    "ElfOnChimp",
    "Elf",
    "BUFF",
    "DeltaSprintz",
    "Zstd",
    "Gzip",
    "Snappy",
    "FFIAlp",
]
skip_methods = set(["RLE"])
for method in skip_methods:
    compression_methods.remove(method)

class SegmentLabelMapping(TypedDict):
    io_read_nanos: List[str]
    io_write_nanos: List[str]
    xor_nanos: List[str]
    others: List[str]
    bit_packing_nanos: List[str]
    decompression_nanos: List[str]
    compare_insert_nanos: List[str]
    delta_nanos: List[str]
    quantization_nanos: List[str]
    sum_nanos: List[str]


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

_hatch_map_exe: dict[str, str] = {}
_color_map_exe: dict[str, tuple[float, float, float, float]] = {}
_color_map: dict[str, tuple[float, float, float, float]] = {}
_hatch_map: dict[str, str] = {}



# Define the JSON file path for storing these mappings
_mappings_file = os.path.join(os.path.dirname(__file__), "common_mappings.json")


def init_mappings() -> None:
    """Initialize global mappings from the JSON file if it exists."""
    global _hatch_map_exe, _color_map_exe, _hatch_map
    if os.path.exists(_mappings_file):
        with open(_mappings_file, "r") as f:
            data = json.load(f)
            _hatch_map_exe = data.get("hatch_map_exe", {})
            _color_map_exe = data.get("color_map_exe", {})
            _color_map = data.get("color_map", {})
            _hatch_map = data.get("hatch_map", {})
    else:
        _hatch_map_exe = {}
        _color_map_exe = {}
        _color_map = {}
        _hatch_map = {}


def update_mappings() -> None:
    """Write the current mappings to the JSON file."""
    data = {
        "hatch_map_exe": _hatch_map_exe,
        "color_map_exe": _color_map_exe,
        "color_map": _color_map,
        "hatch_map": _hatch_map,
    }
    with open(_mappings_file, "w") as f:
        json.dump(data, f, indent=2)


def get_color_exe(label: str) -> tuple[float, float, float, float]:
    import seaborn as sns

    global _color_map_exe
    if label in _color_map_exe:
        return _color_map_exe[label]
    next_index = len(_color_map_exe)
    if next_index >= 20:
        print(f"Adding color for label: {label}, index: {next_index}")
        print(f"Current color map: \n{_color_map_exe}")
    assert next_index < 20
    # color_map_exe[label] = matplotlib.colormaps["tab20"](next_index)
    _color_map_exe[label] = sns.color_palette("tab20", 20)[next_index]
    update_mappings()

    return _color_map_exe[label]

# Call init_mappings once when this module is imported.
init_mappings()

def get_hatch_exe(label: str) -> str:
    global hatch_map_exe
    plt.rcParams["hatch.linewidth"] = 0.3
    hatches = [
        "O.",
        "//",
        "++",
        "//",
        "\\\\",
        "||",
        "--",
        "++",
        "xx",
        "oo",
        "OO",
        "..",
        "**",
    ]
    hatches.reverse()

    if label in _hatch_map_exe:
        return _hatch_map_exe[label]

    next_index = len(_hatch_map_exe) % len(hatches)
    _hatch_map_exe[label] = hatches[next_index]
    update_mappings()

    return _hatch_map_exe[label]


def get_hatch(label: str) -> str:
    global _hatch_map
    plt.rcParams["hatch.linewidth"] = 0.3
    hatches = [
        "O.",
        "//",
        "++",
        "//",
        "\\\\",
        "||",
        "--",
        "++",
        "xx",
        "oo",
        "OO",
        "..",
        "**",
    ]
    hatches.reverse()

    if label in _hatch_map:
        return _hatch_map[label]

    next_index = len(_hatch_map) % len(hatches)
    _hatch_map[label] = hatches[next_index]
    update_mappings()

    return _hatch_map[label]


def get_color(label: str) -> tuple[float, float, float, float]:
    # Define a set of distinct colors for up to 11 methods
    global _color_map
    if label in _color_map:
        return _color_map[label]
    next_index = len(_color_map)
    _color_map[label] = matplotlib.colormaps["tab20"](next_index)

    return _color_map[label]