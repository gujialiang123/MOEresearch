-- Aggregate memcpy by direction.
-- copyKind enum values are in ENUM_CUDA_MEMCPY_OPER; common ones:
--   1 = H2D, 2 = D2H, 3 = D2D, 8 = H2H
SELECT m.copyKind,
       COUNT(*)             AS ops,
       SUM(m.bytes)         AS total_bytes,
       SUM(m.end - m.start) AS total_ns,
       AVG(m.end - m.start) AS avg_ns
FROM CUPTI_ACTIVITY_KIND_MEMCPY m
WHERE (:win_start IS NULL OR m.start >= :win_start)
  AND (:win_end   IS NULL OR m.end   <= :win_end)
GROUP BY m.copyKind;
