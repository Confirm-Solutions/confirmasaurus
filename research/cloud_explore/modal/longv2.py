import time

import modal

stub = modal.Stub("example-train")

N = 3
launch_time = time.time()


@stub.function()
def train():

    if time.time() - launch_time > 60:
        modal.container_app.stop()

    for i in range(10):
        print("training")
        time.sleep(0.5)

    stats = train.get_current_stats()
    jobs_needed = max(N - stats.backlog - stats.num_active_runners, 0)
    for _ in range(jobs_needed):
        train.submit()
