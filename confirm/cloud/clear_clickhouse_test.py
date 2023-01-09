import confirm.cloud.clickhouse as ch
import confirm.cloud.modal_util as modal_util


def clear_ch_dbs():
    client = ch.get_ch_client()
    ch.clear_dbs(client, yes=True)


modal_util.run_on_modal(clear_ch_dbs)
