import confirm.cloud.clickhouse as ch
import confirm.cloud.modal_util as modal_util


def clear_ch_dbs():
    ch_client = ch.get_ch_client()
    redis_client = ch.get_redis_client()
    ch.clear_dbs(ch_client, redis_client, yes=True)


modal_util.run_on_modal(clear_ch_dbs)
