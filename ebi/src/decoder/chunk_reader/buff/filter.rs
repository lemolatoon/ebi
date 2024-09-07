#[cfg(all(target_arch = "x86_64", not(miri)))]
#[inline]
pub(super) fn split_for_simd_processing(
    chunk: &[u8],
    logical_offset: u32,
) -> (&[u8], &[u8], &[u8]) {
    let offset = logical_offset as usize;
    let total_len = chunk.len();

    // 1. Pre-processing: The portion before the offset is aligned to 8 bytes.
    let align_offset = (offset + 7) & !7;
    let initial_len = align_offset.saturating_sub(offset);
    let initial_part = &chunk[..initial_len.min(total_len)];

    // 2. SIMD processing: The portion that is processed in 32-byte units.
    let simd_start = initial_len;
    let simd_len = if simd_start < total_len {
        ((total_len - simd_start) / 32) * 32
    } else {
        0
    };
    let simd_part = if simd_start < total_len {
        &chunk[simd_start..simd_start + simd_len]
    } else {
        &[]
    };

    // 3. Post-processing: The remaining portion.
    let remaining_start = simd_start + simd_len;
    let remaining_part = if remaining_start < total_len {
        &chunk[remaining_start..]
    } else {
        &[]
    };

    (initial_part, simd_part, remaining_part)
}
