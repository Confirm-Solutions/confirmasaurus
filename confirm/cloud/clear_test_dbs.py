import dotenv

import confirm.cloud.clickhouse as ch

dotenv.load_dotenv()
ch_client = ch.get_ch_client()
redis_client = ch.get_redis_client()
ch.clear_dbs(ch_client, redis_client, yes=True, drop_all_redis_keys=True)
