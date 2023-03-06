import time

start = time.time()
import numpy as np  # noqa: E402
import imprint as ip  # noqa: E402
from imprint.models.ztest import ZTest1D  # noqa: E402
import confirm.adagrid as ada  # noqa: E402
import confirm.cloud.clickhouse as ch  # noqa: E402
from confirm.cloud.modal_backend import ModalBackend  # noqa: E402


def main():
    import dotenv

    dotenv.load_dotenv()
    ip.configure_logging()
    db = ch.Clickhouse.connect()
    d = 2
    g = ip.cartesian_grid(
        theta_min=np.full(d, -1),
        theta_max=np.full(d, 0),
        null_hypos=[ip.hypo("theta0 > theta1")],
    )
    _ = ada.ada_validate(
        ZTest1D,
        db=db,
        g=g,
        lam=-1.96,
        prod=False,
        tile_batch_size=1,
        # n_K_double=7,
        # max_target=0.001,
        # global_target=0.002,
        # step_size=2**14,
        # packet_size=2**10,
        # n_steps=30
        backend=ModalBackend(gpu="any"),
    )
    print(db.job_id)


if __name__ == "__main__":
    main()
    print("total runtime", time.time() - start)
