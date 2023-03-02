import modal

import confirm.cloud.modal_util as modal_util

stub = modal.Stub("two_workers")


@stub.function(
    image=modal_util.get_image(dependency_groups=["test", "cloud"]),
    retries=0,
    mounts=modal.create_package_mounts(["confirm", "imprint"]),
    secret=modal.Secret.from_name("confirm-secrets"),
)
def worker(start, job_id=None, n_workers=0):
    import imprint as ip
    import confirm.adagrid as ada
    import confirm.cloud.clickhouse as ch
    from imprint.models.ztest import ZTest1D

    db = ch.Clickhouse.connect(job_id=job_id)
    if start:
        g = ip.cartesian_grid(
            theta_min=[-1], theta_max=[1], null_hypos=[ip.hypo("x0 < 0")]
        )
        reports, _ = ada.ada_calibrate(
            ZTest1D,
            g=g,
            db=db,
            nB=5,
            n_iter=0,
            step_size=32,
            packet_size=8,
            grid_target=0.00015,
            bias_target=0.00015,
        )
        futures = [
            worker.spawn(False, db.job_id, n_workers=0) for _ in range(n_workers)
        ]
    out = [ada.ada_calibrate(ZTest1D, db=db, n_iter=100)[:2]]
    if start:
        for f in futures:
            out += f.get()

        At = db.get_all()
        Bt = ch.Clickhouse.connect(job_id="326be3783f3d42babc373b3163553c02").get_all()
        import pandas as pd

        drop_cols = ["id", "parent_id", "step_iter", "creator_id", "processor_id"]
        pd.testing.assert_frame_equal(
            At.drop(drop_cols, axis=1)
            .sort_values(["step_id", "theta0"])
            .reset_index(drop=True),
            Bt.drop(drop_cols, axis=1)
            .sort_values(["step_id", "theta0"])
            .reset_index(drop=True),
        )

        dup_df = ch._query_df(
            db.client, "select id from tiles group by id having count(*) > 1"
        )
        assert dup_df.shape[0] == 0
    return out


def serial(db="ch"):
    import imprint as ip
    import confirm.adagrid as ada
    import confirm.cloud.clickhouse as ch
    from imprint.models.ztest import ZTest1D

    if db == "ch":
        db = ch.Clickhouse.connect()
    else:
        db = ada.DuckDBTiles.connect()
    g = ip.cartesian_grid(theta_min=[-1], theta_max=[1], null_hypos=[ip.hypo("x0 < 0")])
    reports, _ = ada.ada_calibrate(
        ZTest1D,
        g=g,
        db=db,
        nB=5,
        step_size=32,
        packet_size=8,
        grid_target=0.00015,
        bias_target=0.00015,
    )


def parallel(n_workers=6):
    with stub.run():
        worker.call(True, n_workers=n_workers)


if __name__ == "__main__":
    for i in range(1):
        serial()
        # parallel()
