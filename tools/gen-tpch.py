import duckdb
# from duckdb_extensions import extension_importer

# extension_importer.import_extension("tpch")

conn = duckdb.connect()

conn.execute("INSTALL tpch; LOAD tpch;")
print("tpch installed and loaded.")

conn.execute("CALL dbgen(sf := 10);")

# データ確認
print(conn.execute("SHOW ALL TABLES;").fetchall())
