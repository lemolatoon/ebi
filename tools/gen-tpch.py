import duckdb
import os
import sys

conn = duckdb.connect()
base_dir = "tpch_data"
os.makedirs(base_dir, exist_ok=True)
os.chdir(base_dir)

if len(sys.argv) == 2 and sys.argv[1] == "-r":
    import pyarrow.parquet as pq
    for f in ["h01.parquet", "h06.parquet"]:
        print(f"=== {f} ===")
        table = pq.read_table(f)
        print(table.schema)
    sys.exit(0)
elif len(sys.argv) >= 2:
    print("use '-r' flag to read parquet. or no flag to generate parquet")
    sys.exit(0)

def exec_print(sql: str):
    print(conn.execute(sql).fetchall())

conn.execute("INSTALL tpch; LOAD tpch;")
print("tpch installed and loaded.")

conn.execute("CALL dbgen(sf := 10);")

print(conn.execute("SHOW ALL TABLES;").fetchall())
print("h01:")
print("all executed on DuckDB:")
exec_print("""
SELECT l_returnflag,
       l_linestatus,
       sum(l_quantity) AS sum_qty,
       sum(l_extendedprice) AS sum_base_price,
       sum(l_extendedprice*(1-l_discount)) AS sum_disc_price,
       sum(l_extendedprice*(1-l_discount)*(1+l_tax)) AS sum_charge,
       avg(l_quantity) AS avg_qty,
       avg(l_extendedprice) AS avg_price,
       avg(l_discount) AS avg_disc,
       count(*) AS count_order
FROM lineitem
WHERE l_shipdate <= '1998-09-02'  -- date '1998-12-01' - interval '[DELTA=90]' DAY
GROUP BY l_returnflag,
         l_linestatus
ORDER BY l_returnflag,
         l_linestatus;
""")
print("preprocess in DuckDB:")
exec_print("""
-- h01_duckdb.sql  →  h01.parquet
COPY (
    SELECT                                   -- All DATE/VARCHAR filters already applied
           l_returnflag,
           l_linestatus,
           l_quantity,
           l_extendedprice,
           l_discount,
           l_tax
    FROM lineitem
    WHERE l_shipdate <= DATE '1998-09-02'    -- DuckDB filter (DATE)
) TO 'h01.parquet' (FORMAT PARQUET);
""")
print("please execute operations in EBI:")
print("""
-- h01_ebi.sql  (no GROUP-BY; global roll-up)
SELECT
       SUM(l_quantity)                                        AS sum_qty,
       SUM(l_extendedprice)                                   AS sum_base_price,
       SUM(l_extendedprice * (1 - l_discount))                AS sum_disc_price,
       SUM(l_extendedprice * (1 - l_discount) * (1 + l_tax))  AS sum_charge,
       AVG(l_quantity)                                        AS avg_qty,
       AVG(l_extendedprice)                                   AS avg_price,
       AVG(l_discount)                                        AS avg_disc,
       COUNT(*)                                               AS row_count
FROM read_parquet('h01.parquet');
""")
print("h06")
print("all executed on DuckDB:")
exec_print("""
SELECT sum(l_extendedprice * l_discount) AS revenue
FROM lineitem
WHERE l_shipdate >= '1994-01-01'
  AND l_shipdate < '1995-01-01'
  AND l_discount BETWEEN 0.05 AND 0.07
  AND l_quantity < 24
;
""")
print("preprocess in DuckDB:")
print("all executed on DuckDB:")
exec_print("""
COPY (
    SELECT l_extendedprice, l_discount, l_quantity
    FROM lineitem
    WHERE l_shipdate >= '1994-01-01'
    AND l_shipdate < '1995-01-01'
) TO 'h06.parquet' (FORMAT PARQUET);
""")
print("please execute operations in EBI:")
print("""
SELECT SUM(l_extendedprice * l_discount) AS revenue
WHERE
  AND l_discount BETWEEN 0.05 AND 0.07
  AND l_quantity < 24
FROM read_parquet('h06.parquet');
""")
