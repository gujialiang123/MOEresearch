-- Largest gaps between consecutive kernels on a stream.
-- Requires SQLite window functions (3.25+).
WITH ordered AS (
  SELECT k.start, k.end,
         s.value AS name,
         k.streamId,
         LAG(k.end)   OVER (PARTITION BY k.streamId ORDER BY k.start) AS prev_end,
         LAG(s.value) OVER (PARTITION BY k.streamId ORDER BY k.start) AS prev_name
  FROM CUPTI_ACTIVITY_KIND_KERNEL k
  JOIN StringIds s ON k.shortName = s.id
  WHERE (:stream_id IS NULL OR k.streamId = :stream_id)
    AND (:win_start IS NULL OR k.start >= :win_start)
    AND (:win_end   IS NULL OR k.end   <= :win_end)
)
SELECT prev_name AS before_kernel,
       name      AS after_kernel,
       streamId,
       prev_end  AS gap_start_ns,
       start     AS gap_end_ns,
       (start - prev_end) AS gap_ns
FROM ordered
WHERE prev_end IS NOT NULL AND (start - prev_end) > 0
ORDER BY gap_ns DESC
LIMIT :top_n;
