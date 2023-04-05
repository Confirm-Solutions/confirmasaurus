from typing import List

import dotenv
import typer

import confirm.cloud.clickhouse as ch


def main(
    prefix: str = "unnamed",
    y: bool = False,
    list: bool = False,
    service: str = "TEST",
    exclude: List[str] = None,
):
    dotenv.load_dotenv()
    ch_client = ch.get_ch_client(service=service)
    ch.clear_dbs(
        ch_client, list=list, prefix=prefix, names=None, yes=y, exclude=exclude
    )


if __name__ == "__main__":
    typer.run(main)
