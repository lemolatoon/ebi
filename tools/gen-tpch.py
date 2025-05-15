import duckdb
from duckdb_extensions import extension_importer

# tpch 拡張をインポート
extension_importer.import_extension("tpch")

conn = duckdb.connect()

# すぐに dbgen を呼べる
conn.execute("CALL dbgen(sf := 10);")

# データ確認
print(conn.execute("SHOW ALL TABLES;").fetchall())
