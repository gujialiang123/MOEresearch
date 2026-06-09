-- Top-N kernels by total self time on a given stream (or all streams if NULL).
-- Bind variables: :stream_id (int or NULL), :top_n (int), :win_start (int or NULL), :win_end (int or NULL)
-- Returns short_name + sample demangled_name so downstream categorisers can
-- distinguish kernels that share a short name (e.g. CUTLASS device_kernel<flash::*>
-- and device_kernel<gemm::*> both report short_name='device_kernel').
SELECT
    s.value AS short_name,
    MAX(ds.value) AS demangled_sample,
    SUM(k.end - k.start) AS self_ns,
    COUNT(*)             AS calls,
    AVG(k.end - k.start) AS avg_ns,
    MAX(k.end - k.start) AS max_ns,
    MAX(k.registersPerThread) AS max_reg,
    MAX(k.gridX) AS max_grid_x, MAX(k.gridY) AS max_grid_y, MAX(k.gridZ) AS max_grid_z,
    MAX(k.blockX) AS max_block_x, MAX(k.blockY) AS max_block_y, MAX(k.blockZ) AS max_block_z
FROM CUPTI_ACTIVITY_KIND_KERNEL k
JOIN StringIds s ON k.shortName = s.id
LEFT JOIN StringIds ds ON k.demangledName = ds.id
WHERE (:stream_id IS NULL OR k.streamId = :stream_id)
  AND (:win_start  IS NULL OR k.start >= :win_start)
  AND (:win_end    IS NULL OR k.end   <= :win_end)
GROUP BY s.value
ORDER BY self_ns DESC
LIMIT :top_n;
