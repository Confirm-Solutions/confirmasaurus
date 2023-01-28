import time

import modal
import pandas as pd

import confirm.cloud.modal_util as modal_util

stub = modal.Stub("duckdbtest", image=modal_util.get_image())

volume = modal.SharedVolume().persist("imprint-db")


@stub.function(shared_volumes={"/data": volume})
def create_table(db_path, df):
    import duckdb

    conn = duckdb.connect(db_path)
    conn.execute("create table tiles as select * from df")
    conn.execute("drop table tiles")
    conn.execute("create table tiles as select * from df")


@stub.function(shared_volumes={"/data": volume})
def get_n_rows(db_path):
    import duckdb

    conn = duckdb.connect(db_path)
    return conn.execute("select count(*) from tiles").fetchall()[0][0]


@stub.function(shared_volumes={"/data": volume})
def get_next(db_path):
    import duckdb

    conn = duckdb.connect(db_path)
    return conn.execute("select * from tiles order by orig_lam limit 10000").fetchall()[
        0
    ][0]


def main():
    df = pd.read_parquet("research/cloud_explore/clickhouse/dbtestsmall.parquet")
    print(df.shape)
    # start = time.time()
    # create_table.get_raw_f()("test.db", df)
    # print(time.time() - start)
    # start = time.time()
    # print(get_n_rows.get_raw_f()("test.db"))
    # print(time.time() - start)
    with stub.run():
        for i in range(40, 45):
            start = time.time()
            create_table(f"/data/test{i}.db", df[:100000])
            print("insert 100k ", time.time() - start)
            start = time.time()
            print(get_n_rows(f"/data/test{i}.db"))
            print("n_rows", time.time() - start)
            start = time.time()
            print(get_next(f"/data/test{i}.db"))
            print("next 10k ", time.time() - start)


if __name__ == "__main__":
    main()
