import time

start = time.time()
import numpy as np  # noqa: E402
import imprint as ip  # noqa: E402
from imprint.models.ztest import ZTest1D  # noqa: E402
import confirm.adagrid as ada  # noqa: E402
from confirm.cloud.modal_backend import ModalBackend  # noqa: E402


def main():
    import dotenv

    dotenv.load_dotenv()
    ip.configure_logging()
    d = 3
    g = ip.cartesian_grid(
        theta_min=np.full(d, -1),
        theta_max=np.full(d, 0),
        null_hypos=[ip.hypo("theta0 > theta1")],
    )
    db = ada.ada_validate(
        ZTest1D,
        g=g,
        lam=-1.96,
        prod=False,
        n_K_double=7,
        max_target=0.001,
        global_target=0.002,
        step_size=2**17,
        packet_size=2**13,
        n_steps=15,
        n_zones=4,
        # backend=CoiledBackend()
        backend=ModalBackend(n_workers=1, gpu="any"),
    )
    report_df = db.get_reports()  # noqa
    print("hi")


if __name__ == "__main__":
    main()
    print("total runtime", time.time() - start)
