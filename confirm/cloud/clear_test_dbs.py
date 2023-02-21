import dotenv

import confirm.cloud.clickhouse as ch

dotenv.load_dotenv()
ch_client = ch.get_ch_client()
ch.clear_dbs(ch_client, yes=True)
