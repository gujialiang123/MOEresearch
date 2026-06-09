-- Aggregate CUDA API call counts and total time.
-- Useful for: launch overhead diagnosis, cudagraph vs eager ratio.
SELECT s.value AS api_name,
       COUNT(*)               AS calls,
       SUM(r.end - r.start)   AS total_ns,
       AVG(r.end - r.start)   AS avg_ns
FROM CUPTI_ACTIVITY_KIND_RUNTIME r
JOIN StringIds s ON r.nameId = s.id
WHERE (:win_start IS NULL OR r.start >= :win_start)
  AND (:win_end   IS NULL OR r.end   <= :win_end)
GROUP BY s.value
ORDER BY calls DESC;
