import modal

stub = modal.Stub("example-train")

N = 3


@stub.function()
def train():
    import time

    for i in range(10):
        print("training")
        time.sleep(0.5)


@stub.function(schedule=modal.Period(minutes=1))
def controller():
    stats = train.get_current_stats()

    # Dummy condition
    if stats.num_active_runners >= N:
        modal.container_app.stop()

    jobs_needed = max(N - stats.backlog - stats.num_active_runners, 0)

    for _ in range(jobs_needed):
        train.submit()
