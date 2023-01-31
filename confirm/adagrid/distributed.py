import logging
import time

from .convergence import WorkerStatus

logger = logging.getLogger(__name__)


def run(algo, db, report, n_steps):
    """
    One iteration of the adagrid algorithm.

    Parallelization:
    There are four main portions of the adagrid algo:
    - Convergence criterion --> serialize to avoid data races.
    - Tile selection --> serialize to avoid data races.
    - Deciding whether to refine or deepen --> parallelize.
    - Simulation and CSE --> parallelize.
    By using a distributed lock around the convergence/tile selection, we can have
    equality between workers and not need to have a leader node.

    Args:
        i: The iteration number.

    Returns:
        done: A value from the Enum Convergence:
             (INCOMPLETE, CONVERGED, FAILED).
        report: A dictionary of information about the iteration.
    """

    start = time.time()

    ########################################
    # Step 1: Get the next batch of tiles to process. We do this before
    # checking for convergence because part of the convergence criterion is
    # whether there are any impossible tiles.
    ########################################
    max_loops = 25
    i = 0
    status = WorkerStatus.WORK
    report["runtime_wait_for_lock"] = 0
    report["runtime_wait_for_work"] = 0
    while i < max_loops:
        logger.debug("Starting loop %s.", i)
        report["waitings"] = i
        with db.lock:
            logger.debug("Claimed DB lock.")
            report["runtime_wait_for_lock"] += time.time() - start
            start = time.time()

            step_id, step_iter, step_n_iter, step_n_tiles = db.get_step_info()
            report["step_id"] = step_id
            report["step_iter"] = step_iter
            report["step_n_iter"] = step_n_iter
            report["step_n_tiles"] = step_n_tiles

            # Check if there are iterations left in this step.
            # If there are, get the next batch of tiles to process.
            if step_iter < step_n_iter:
                logger.debug("get_work(step_id=%s, step_iter=%s)", step_id, step_iter)
                work = db.get_work(step_id, step_iter)
                report["runtime_get_work"] = time.time() - start
                start = time.time()
                report["work_extraction_time"] = time.time()
                report["n_processed"] = work.shape[0]
                logger.debug("get_work(...) returned %s tiles.", work.shape[0])

                # If there's work, return it!
                if work.shape[0] > 0:
                    db.set_step_info(
                        step_id=step_id,
                        step_iter=step_iter + 1,
                        n_iter=step_n_iter,
                        n_tiles=step_n_tiles,
                    )
                    logger.debug("Processing %s tiles.", work.shape[0])
                    report["runtime_update_step_info"] = time.time() - start
                    start = time.time()
                    results = algo.process_tiles(tiles_df=work, report=report)
                    report["runtime_processing"] = time.time() - start
                    db.insert_results(results)
                    return status, report
                else:
                    # If step_iter < step_n_iter but there's no work, then
                    # The INSERT into tiles that was supposed to populate
                    # the work is probably incomplete. We should wait a
                    # very short time and try again.
                    logger.debug("No work despite step_iter < step_n_iter.")
                    wait = 0.1

            # If there are no iterations left in the step, we check if the
            # step is complete. For a step to be complete means that all
            # tiles have results.
            else:
                n_processed_tiles = db.n_processed_tiles(step_id)
                report["n_finished_tiles"] = n_processed_tiles
                if n_processed_tiles == step_n_tiles:
                    # If a packet has just been completed, we check for convergence.
                    status = algo.convergence_criterion(report=report)
                    report["runtime_convergence_criterion"] = time.time() - start
                    start = time.time()
                    if status:
                        logger.debug("Convergence!!")
                        return WorkerStatus.CONVERGED, report

                    if step_id >= n_steps - 1:
                        # We've completed all the steps, so we're done.
                        logger.debug("Reached max number of steps. Terminating.")
                        return WorkerStatus.REACHED_N_STEPS, report

                    # If we haven't converged, we create a new step.
                    new_step_id = algo.new_step(step_id + 1, report=report)

                    report["runtime_new_step"] = time.time() - start
                    start = time.time()
                    if new_step_id == "empty":
                        # New packet is empty so we have terminated but
                        # failed to converge.
                        logger.debug(
                            "New packet is empty. Terminating despite "
                            "failure to converge."
                        )
                        return WorkerStatus.FAILED, report
                    else:
                        # Successful new packet. We should check for work again
                        # immediately.
                        status = WorkerStatus.NEW_STEP
                        wait = 0
                        logger.debug("Successfully created new packet.")
                else:
                    # No work available, but the packet is incomplete. This is
                    # because other workers have claimed all the work but have not
                    # processsed yet.
                    # In this situation, we should release the lock and wait for
                    # other workers to finish.
                    wait = 1
                    logger.debug("No work available, but packet is incomplete.")
        if wait > 0:
            logger.debug("Waiting %s seconds and checking for work again.", wait)
            time.sleep(wait)
        if i > 3:
            logger.warning(
                "Worker s has been waiting for work for"
                " %s iterations. This might indicate a bug.",
                i,
            )
        report["runtime_wait_for_work"] += time.time() - start
        i += 1

    return WorkerStatus.STUCK, report
