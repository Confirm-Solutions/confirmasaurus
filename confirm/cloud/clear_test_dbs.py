from pprint import pprint

import dotenv
import typer

import confirm.cloud.clickhouse as ch


def main(prefix: str = "unnamed", y: bool = False, list: bool = False):
    dotenv.load_dotenv()
    ch_client = ch.get_ch_client()
    if list:
        print("All databases: ")
        pprint(ch.list_dbs(ch_client))
    ch.clear_dbs(ch_client, prefix=prefix, names=None, yes=y)


if __name__ == "__main__":
    typer.run(main)
