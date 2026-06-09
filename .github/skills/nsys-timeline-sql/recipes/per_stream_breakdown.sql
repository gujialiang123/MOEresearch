-- GPU active time + kernel count per stream.
-- Active = sum of kernel durations (no double-counting since one kernel per stream at a time).
-- Idle = (max_end - min_start) - active, per stream.
SELECT k.streamId,
       COUNT(*)                  AS kernel_count,
       SUM(k.end - k.start)      AS active_ns,
       MIN(k.start)              AS first_kernel_ns,
       MAX(k.end)                AS last_kernel_ns,
       (MAX(k.end) - MIN(k.start)) - SUM(k.end - k.start) AS idle_ns
FROM CUPTI_ACTIVITY_KIND_KERNEL k
WHERE (:win_start IS NULL OR k.start >= :win_start)
  AND (:win_end   IS NULL OR k.end   <= :win_end)
GROUP BY k.streamId
ORDER BY active_ns DESC;
