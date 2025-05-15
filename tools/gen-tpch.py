import duckdb
import os

conn = duckdb.connect()
base_dir = "tpch_data"
os.makedirs(base_dir, exist_ok=True)
os.chdir(base_dir)

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
           l_quantity,
           l_extendedprice,
           l_discount,
           l_tax
    FROM lineitem
    WHERE l_shipdate <= DATE '1998-09-02'    -- DuckDB filter (DATE)
) TO 'h01.parquet' (FORMAT PARQUET);
""")
print("please execute operations in EBI:")
"""
-- h01_ebi.sql  (no GROUP-BY; global roll-up)
SELECT
       SUM(l_quantity)                                        AS sum_qty,
       SUM(l_extendedprice)                                   AS sum_base_price,
       SUM(l_extendedprice * (1 - l_discount))                AS sum_disc_price,
       SUM(l_extendedprice * (1 - l_discount) * (1 + l_tax))  AS sum_charge,
       AVG(l_quantity)                                        AS avg_qty,
       AVG(l_extendedprice)                                   AS avg_price,
       AVG(l_discount)                                        AS avg_disc,
       SUM(1.0)                                               AS row_count   -- COUNT emulation
FROM read_parquet('h01.parquet');
"""
