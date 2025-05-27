#!/bin/bash
#
# USAGE:
#   ./plot_all.sh <final_save_directory>
#
# PARAMETERS:
#   <final_save_directory>
#       This is the directory where all output files and directories from the plotting scripts
#       will be consolidated. The script will create this directory if it does not exist.
#
# ENVIRONMENT VARIABLES:
#   TPCH_DIR:
#       Directory containing TPCH data (default: ../chameleon/tpch002)
#
#   XOR_JSON:
#       Path to the XOR JSON file (default: ../chameleon/xor/000.json)
#
#   EMBEDDING_JSON:
#       Path to the EMBEDDING JSON file (default: ../chameleon/embedding/embedding_result.json)
#
#   UCR2018_DIR:
#       Directory for UCR2018 related files (default: ../chameleon/ucr2018)
#
#   MATRIX_CUDA_DIR:
#       Directory containing MATRIX_CUDA data (default: ../chameleon/matrix_cuda/000)
#
#   ALL_AVX2_DISK_JSON:
#       Path to the AVX2 DISK JSON file (default: ../chameleon/all/haswell_disk2/000.json)
#
#   ALL_AVX2_MEMORY_JSON:
#       Path to the AVX2 MEMORY JSON file (default: ../chameleon/all/haswell_memory/000.json)
#
#   ALL_AVX512_DISK_JSON:
#       Path to the AVX512 DISK JSON file (default: ../chameleon/all/cascade_native_disk2/001.json)
#
#   ALL_AVX512_MEMORY_JSON:
#       Path to the AVX512 MEMORY JSON file (default: ../chameleon/all/cascade_native_memory/001.json)
#
# FINAL SAVE DIRECTORY STRUCTURE:
#   The provided final_save_directory will have the following structure:
#     - tpch/            : Output from plot_tpch.py (copied from ./results/tpch)
#     - matmul_cuda/     : Output from matrix_plot2.py (copied from ./results/matmul_cuda)
#     - avx2_disk/       : Output from plot2.py when run using ALL_AVX2_DISK_JSON (destination subdirectory named "avx2_disk")
#     - avx2_memory/     : Output from plot2.py when run using ALL_AVX2_MEMORY_JSON (destination subdirectory named "avx2_memory")
#     - avx512_disk/     : Output from plot2.py when run using ALL_AVX512_DISK_JSON (destination subdirectory named "avx512_disk")
#     - avx512_memory/   : Output from plot2.py when run using ALL_AVX512_MEMORY_JSON (destination subdirectory named "avx512_memory")
#     - meta.json        : A file recording the environment variable values, input data paths, and a timestamp.
#
# EXAMPLES:
#   ./plot_all.sh ./final_results
#       Runs all the plotting scripts and copies their outputs into "./final_results",
#       following the structure described above.
#
if [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
    sed -n '3,52p' "$0" | sed 's/^#//'
    exit 0
fi
if [ -z "$1" ]; then
    echo "Usage: $0 <final_save_directory>"
    exit 1
fi
FINAL_SAVE_DIR="$1"
mkdir -p "$FINAL_SAVE_DIR"

if [ -z "$TPCH_DIR"]; then
    TPCH_DIR=../chameleon/tpch002
fi
if [ -z "$XOR_JSON" ]; then
    XOR_JSON=../chameleon/xor/000.json
fi
if [ -z "$EMBEDDING_JSON" ]; then
    EMBEDDING_JSON=../chameleon/embedding/embedding_result.json
fi
if [ -z "$UCR2018_DIR" ]; then
    UCR2018_DIR=../chameleon/ucr2018
fi
if [ -z "$MATRIX_CUDA_DIR" ]; then
    MATRIX_CUDA_DIR=../chameleon/matrix_cuda/000
fi
if [ -z "$ALL_AVX2_DISK_JSON" ]; then
    ALL_AVX2_DISK_JSON=../chameleon/all/haswell_disk2/000.json
fi
if [ -z "$ALL_AVX2_MEMORY_JSON" ]; then
    ALL_AVX2_MEMORY_JSON=../chameleon/all/haswell_memory/000.json
fi
if [ -z "$ALL_AVX512_DISK_JSON" ]; then
    ALL_AVX512_DISK_JSON=../chameleon/all/cascade_native_disk2/001.json
fi
if [ -z "$ALL_AVX512_MEMORY_JSON" ]; then
    ALL_AVX512_MEMORY_JSON=../chameleon/all/cascade_native_memory/001.json
fi

# Run plot2.py for each JSON.
# plot2.py creates its output directory in the same directory as the JSON file,
# using the JSON filename without the .json extension.
create_and_copy_plot2() {
    json_path="$1"
    dest_dir_name="$2"
    # Get the directory and base name without '.json'
    json_dir=$(dirname "$json_path")
    base_name=$(basename "$json_path" .json)
    # Run plot2.py; the script will create the output directory: <json_dir>/<base_name>
    uv run plot2.py "$json_path" "$XOR_JSON" "$UCR2018_DIR" "$EMBEDDING_JSON"
    # Use a descriptive variable for the generated output directory.
    plot_output_dir="${json_dir}/${base_name}"
    if [ -d "$plot_output_dir" ]; then
        cp -r "$plot_output_dir" "$FINAL_SAVE_DIR/$dest_dir_name"
    else
        echo "Warning: Expected output directory $plot_output_dir not found for $json_path"
    fi
}

# Run plot_tpch.py.
# It always saves results into "./results/tpch".
uv run plot_tpch.py $TPCH_DIR -y

# Run matrix_plot2.py.
# It always saves results into "./results/matmul_cuda".
uv run matrix_plot2.py $MATRIX_CUDA_DIR -cuda

# For AVX2 Disk
create_and_copy_plot2 "$ALL_AVX2_DISK_JSON" "avx2_disk"
# For AVX2 Memory
create_and_copy_plot2 "$ALL_AVX2_MEMORY_JSON" "avx2_memory"
# For AVX512 Disk
create_and_copy_plot2 "$ALL_AVX512_DISK_JSON" "avx512_disk"
# For AVX512 Memory
create_and_copy_plot2 "$ALL_AVX512_MEMORY_JSON" "avx512_memory"

# After running all scripts, copy the fixed output directories to FINAL_SAVE_DIR.
# Copy plot_tpch.py outputs (./results/tpch).
if [ -d "./results/tpch" ]; then
    cp -r ./results/tpch "$FINAL_SAVE_DIR/tpch"
else
    echo "Warning: ./results/tpch not found."
fi

# Copy matrix_plot2.py outputs (./results/matmul_cuda).
if [ -d "./results/matmul_cuda" ]; then
    cp -r ./results/matmul_cuda "$FINAL_SAVE_DIR/matmul_cuda"
else
    echo "Warning: ./results/matmul_cuda not found."
fi

# Create meta.json in FINAL_SAVE_DIR.
# This file records the directories and files used to generate the results.
cat <<EOF > "$FINAL_SAVE_DIR/meta.json"
{
  "TPCH_DIR": "$TPCH_DIR",
  "XOR_JSON": "$XOR_JSON",
  "EMBEDDING_JSON": "$EMBEDDING_JSON",
  "UCR2018_DIR": "$UCR2018_DIR",
  "MATRIX_CUDA_DIR": "$MATRIX_CUDA_DIR",
  "ALL_AVX2_DISK_JSON": "$ALL_AVX2_DISK_JSON",
  "ALL_AVX2_MEMORY_JSON": "$ALL_AVX2_MEMORY_JSON",
  "ALL_AVX512_DISK_JSON": "$ALL_AVX512_DISK_JSON",
  "ALL_AVX512_MEMORY_JSON": "$ALL_AVX512_MEMORY_JSON",
  "timestamp": "$(date --iso-8601=seconds)"
}
EOF

echo "All results have been copied to $FINAL_SAVE_DIR"
