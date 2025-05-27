#!/bin/bash
#
# USAGE:
#   ./plot_all.sh <final_save_directory> [mode]
#
# PARAMETERS:
#   <final_save_directory>
#       Directory where all output from the plotting scripts will be consolidated.
#       Created if it does not exist.
#
#   [mode]
#       Select which plots to generate. One of:
#         all     Run TPCH, matmul and plot2 (default)
#         tpch    Run only TPCH plots
#         matmul  Run only matmul_cuda plots
#         plot2   Run only plot2 (AVX2/AVX512) plots
#
# ENVIRONMENT VARIABLES:
#   TPCH_DIR             Directory containing TPCH data (default: ../chameleon/tpch002)
#   XOR_JSON             Path to XOR JSON file (default: ../chameleon/xor/000.json)
#   EMBEDDING_JSON       Path to EMBEDDING JSON file (default: ../chameleon/embedding/embedding_result.json)
#   UCR2018_DIR          Directory for UCR2018 files (default: ../chameleon/ucr2018)
#   MATRIX_CUDA_DIR      Directory containing MATRIX_CUDA data (default: ../chameleon/matrix_cuda/000)
#   ALL_AVX2_DISK_JSON   Path to AVX2 DISK JSON (default: ../chameleon/all/haswell_disk2/000.json)
#   ALL_AVX2_MEMORY_JSON Path to AVX2 MEMORY JSON (default: ../chameleon/all/haswell_memory/000.json)
#   ALL_AVX512_DISK_JSON Path to AVX512 DISK JSON (default: ../chameleon/all/cascade_native_disk2/001.json)
#   ALL_AVX512_MEMORY_JSON Path to AVX512 MEMORY JSON (default: ../chameleon/all/cascade_native_memory/001.json)
#
# FINAL SAVE DIRECTORY STRUCTURE:
#   <final_save_directory>/
#     ├─ tpch/           Output from plot_tpch.py
#     ├─ matmul_cuda/    Output from matrix_plot2.py
#     ├─ avx2_disk/      Output from plot2.py using ALL_AVX2_DISK_JSON
#     ├─ avx2_memory/    Output from plot2.py using ALL_AVX2_MEMORY_JSON
#     ├─ avx512_disk/    Output from plot2.py using ALL_AVX512_DISK_JSON
#     ├─ avx512_memory/  Output from plot2.py using ALL_AVX512_MEMORY_JSON
#     └─ meta.json       JSON with input paths and timestamp
#
# EXAMPLES:
#   ./plot_all.sh ./final   # run everything, save into ./final
#   ./plot_all.sh ./final tpch
#   ./plot_all.sh ./final matmul
# END OF HELP

if [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
    sed -n '/^# USAGE:/,/^# END OF HELP/ {
        /^# END OF HELP/d
        s/^# \(.*\)$/\1/
        s/^#$//
        p
    }' "$0"
    exit 0
fi

if [ -z "$1" ]; then
    echo "Error: final_save_directory is required."
    echo "Usage: $0 <final_save_directory> [mode]"
    exit 1
fi

FINAL_SAVE_DIR="$1"
MODE="${2:-all}"    # default mode is "all"
mkdir -p "$FINAL_SAVE_DIR"

# Set defaults for environment variables if unset
: "${TPCH_DIR:=../chameleon/tpch002}"
: "${XOR_JSON:=../chameleon/xor/000.json}"
: "${EMBEDDING_JSON:=../chameleon/embedding/embedding_result.json}"
: "${UCR2018_DIR:=../chameleon/ucr2018}"
: "${MATRIX_CUDA_DIR:=../chameleon/matrix_cuda/000}"
: "${ALL_AVX2_DISK_JSON:=../chameleon/all/haswell_disk2/000.json}"
: "${ALL_AVX2_MEMORY_JSON:=../chameleon/all/haswell_memory/000.json}"
: "${ALL_AVX512_DISK_JSON:=../chameleon/all/cascade_native_disk2/001.json}"
: "${ALL_AVX512_MEMORY_JSON:=../chameleon/all/cascade_native_memory/001.json}"

#######################################
# Function: create_and_copy_plot2
#######################################
create_and_copy_plot2() {
    local json_path="$1" dest="$2"
    local json_dir=$(dirname "$json_path")
    local base=$(basename "$json_path" .json)
    uv run plot2.py "$json_path" "$XOR_JSON" "$UCR2018_DIR" "$EMBEDDING_JSON"
    uv run plot_cd.py "$json_path"
    local out_dir="${json_dir}/${base}"
    if [ -d "$out_dir" ]; then
        cp -r "$out_dir" "$FINAL_SAVE_DIR/$dest"
    else
        echo "Warning: expected $out_dir not found"
    fi
}

#######################################
# Function: run_tpch
#######################################
run_tpch() {
    uv run plot_tpch.py "$TPCH_DIR" -y
    if [ -d "./results/tpch" ]; then
        cp -r "./results/tpch" "$FINAL_SAVE_DIR/tpch"
    else
        echo "Warning: ./results/tpch not found"
    fi
}

#######################################
# Function: run_matmul
#######################################
run_matmul() {
    uv run matrix_plot2.py "$MATRIX_CUDA_DIR" -cuda
    if [ -d "./results/matmul_cuda" ]; then
        cp -r "./results/matmul_cuda" "$FINAL_SAVE_DIR/matmul_cuda"
    else
        echo "Warning: ./results/matmul_cuda not found"
    fi
}

run_embedding() {
    uv run embedding_plot.py "$EMBEDDING_JSON"
    if [ -d "./results/embedding" ]; then
        cp -r "./results/embedding" "$FINAL_SAVE_DIR/embedding"
    else
        echo "Warning: ./results/embedding not found"
    fi
}

run_ucr2018() {
    uv run ucr2018_plot.py "$UCR2018_DIR"
    if [ -d "./results/ucr2018" ]; then
        cp -r "./results/ucr2018" "$FINAL_SAVE_DIR/ucr2018"
    else
        echo "Warning: ./results/ucr2018 not found"
    fi
}

#######################################
# Function: run_plot2_all
#######################################
run_plot2_all() {
    create_and_copy_plot2 "$ALL_AVX2_DISK_JSON" "avx2_disk"
    create_and_copy_plot2 "$ALL_AVX2_MEMORY_JSON" "avx2_memory"
    create_and_copy_plot2 "$ALL_AVX512_DISK_JSON" "avx512_disk"
    create_and_copy_plot2 "$ALL_AVX512_MEMORY_JSON" "avx512_memory"
}

#######################################
# Main Execution Dispatcher
#######################################
case "$MODE" in
    all)
        run_tpch
        run_matmul
        run_plot2_all
        run_embedding
        run_ucr2018
        ;;
    tpch)
        run_tpch
        ;;
    matmul)
        run_matmul
        ;;
    plot2)
        run_plot2_all
        run_embedding
        run_ucr2018
        ;;
    *)
        echo "Error: unknown mode '$MODE'. Use one of all|tpch|matmul|plot2"
        exit 1
        ;;
esac

# Update meta.json: preserve previous entries and update only current mode keys
meta_file="$FINAL_SAVE_DIR/meta.json"
tmp_meta_file=$(mktemp)

if [ -f "$meta_file" ]; then
    cp "$meta_file" "$tmp_meta_file"
else
    echo "{}" > "$tmp_meta_file"
fi

# Function to update a key in the JSON
update_meta() {
    key="$1"
    value="$2"
    jq --arg k "$key" --arg v "$value" '.[$k] = $v' "$tmp_meta_file" > "$tmp_meta_file.tmp" && mv "$tmp_meta_file.tmp" "$tmp_meta_file"
}

# Always update timestamp
update_meta "timestamp" "$(date --iso-8601=seconds)"

# Update only keys related to the executed mode
case "$MODE" in
  all)
    update_meta "TPCH_DIR" "$TPCH_DIR"
    update_meta "MATRIX_CUDA_DIR" "$MATRIX_CUDA_DIR"
    update_meta "XOR_JSON" "$XOR_JSON"
    update_meta "EMBEDDING_JSON" "$EMBEDDING_JSON"
    update_meta "UCR2018_DIR" "$UCR2018_DIR"
    update_meta "ALL_AVX2_DISK_JSON" "$ALL_AVX2_DISK_JSON"
    update_meta "ALL_AVX2_MEMORY_JSON" "$ALL_AVX2_MEMORY_JSON"
    update_meta "ALL_AVX512_DISK_JSON" "$ALL_AVX512_DISK_JSON"
    update_meta "ALL_AVX512_MEMORY_JSON" "$ALL_AVX512_MEMORY_JSON"
    ;;
  tpch)
    update_meta "TPCH_DIR" "$TPCH_DIR"
    ;;
  matmul)
    update_meta "MATRIX_CUDA_DIR" "$MATRIX_CUDA_DIR"
    ;;
  plot2)
    update_meta "XOR_JSON" "$XOR_JSON"
    update_meta "EMBEDDING_JSON" "$EMBEDDING_JSON"
    update_meta "UCR2018_DIR" "$UCR2018_DIR"
    update_meta "ALL_AVX2_DISK_JSON" "$ALL_AVX2_DISK_JSON"
    update_meta "ALL_AVX2_MEMORY_JSON" "$ALL_AVX2_MEMORY_JSON"
    update_meta "ALL_AVX512_DISK_JSON" "$ALL_AVX512_DISK_JSON"
    update_meta "ALL_AVX512_MEMORY_JSON" "$ALL_AVX512_MEMORY_JSON"
    ;;
esac

# Save final version
mv "$tmp_meta_file" "$meta_file"

echo "Completed mode='$MODE'. Results collected in '$FINAL_SAVE_DIR'."
