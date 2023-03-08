# Running non-adagrid jobs using the adagrid code.

It is possible to run calibration or validation *without adagrid* using the
interface provided by ada_validate and ada_calibrate! In order to do these,
follow the examples given in test_calibration_nonadagrid_using_adagrid and
test_validation_nonadagrid_using_adagrid. This is useful for two reasons:
- we can use the database backend.
- we can run distributed non-adagrid jobs when we want more happy more
  computers.
  
# Distributed adagrid and database backend

## Design Principles and goals:
- Simplicity.
- Append-only and no updates to the extent possible. This makes the "story" of
  an adagrid run very clear after the fact which is useful both for
  replicability and debugging.
    - Additional storyline is stored in the `reports` and `logs` tables.
- Results are written to DB ASAP. This avoids data loss.
- Minimize communication overhead, maximize performance! Keep between-worker
  communication isolated to the coordination stage.
- Assume that the DB does not lose inserts.
- Good to separate computation from communications.
- Idempotency is an important property of each stage in the pipeline. --> we
  should be able to kill any stage at any point and then re-run it and get the
  same answer.
- Verify system properties often. For example, there should be a defined
  relationship between the `results` and `tiles` and `done`.

## Adagrid algorithm high-level overview.

Leader node: Where the job is launched. This could be your local machine but in
larger jobs, it will probably a CPU-only instance.
Worker node: Runs steps and processes/simulates packets.

The adagrid algorithm consists of global "coordinations" and then "steps"
within each zone.
1. A "coordination" - The leader node re-assigns all eligible tiles to zones. 
    - An "eligible" tile is one that could be refined or deepened. 
    - A "zone" is a collection of tiles that can be worked on by a worker.
    - We separate the domain into these zones so that workers can operate
      indendently.
2. A "step" occurs on a worker and operates on a zone. Steps 1-2 are included
   in the `new_step` function and step 3 is `process_packet`.
    1. Convergence critierion: The worker decides whether we've reached
       convergence in this zone.
    2. Tile selection and creation: If no convergence, we choose and then
       refine or deepen some tiles in this zone. Both the children of a refined
       tile and the deepened tile with higher `n_sims` will be considered new
       tiles. Those new tiles will be split into several "packets". 
    3. Simulation: *This is the expensive portion of the algorithm!* Each packet
       will be launched and simulated on a worker. After packets have all been
       processed, there will be no tiles without results.
       
Notes:
- in theory, the algorithm can stop or be killed during any step and be
  restarted. In practice, this is a work in progress.
- everything in the process above is determinstic as a function of a few inputs:
    1. Model seed - used to generate simulation inputs
    2. Bootstrapping seed - used to generate resamplings for calibration.
    3. The number of zones. 
- an example step might refine 10k tiles in 2D, resulting in 40k new
  tiles. These might get split into 8 packets with 5,000 packets each. 
- within a packet, we also further subdivide the simulations into batches.
  Typically these might be 10s or 100s of tiles. The purpose of this batching
  is to reduce memory requirements. Simulating 100 tiles with 100k simulations
  per tile with bootstrap calibration will result in 100 * 100 * 100000
  simulation outputs. At 32 bits for each output, that is 4 GB of data. These
  results are then condensed down to just 100 results per tile because we are
  interested in the rate of Type I Errors and not the particular circumstances
  of each Type I Error.
  
## "Algorithms" for Adagrid

There are two "algorithms" using the adagrid framework:
- `AdaValidate` - validation!
    - Both tile selection and convergence depend on two parameters:
        - `max_target`: The limit on allowed slack in fraction Type I Error for
          tiles that have a Type I Error bound that is above the worst case
          simulated Type I Error. This convergence criterion parameter is
          useful for tightening the bound in the areas where it matters most.
          Defaults to 0.002.
        - `global_target`: The limit on allowed slack in fraction Type I Error
          for any tile regardless of its Type I Error bound value. This
              convergence criterion parameter is useful for getting a tighter
              fit to the Type I Error surface throughout parameter space
              regardless of the value of the Type I Error. Defaults to 0.005.
    - Tile selection refines/deepens tiles that violate the `max_target` or
      `global_target` tolerances. Tiles are refined based on a heuristic that
      compares what portion of the tile's slack is consumed by CSE versus
      Clopper-Pearson.

- `AdaCalibrate` - calibration!
    - The convergence criterion here depends on both the grid cost and the bias
      of $\lambda^{**}$. 
        - The bias is estimated by bootstrapping the $\lambda^{*}$ at each
          tile.
        - The grid cost is defined as $\alpha - \alpha_0$ where $\alpha_0 =
          \mathrm{InverseCSE}(\alpha, \mathrm{this ~ tile})$. This is
          essentially how much slack is going to continuous simulation
          extension.
    - Tile selection:
        - The tile selection is not a solved problem 
        - Currently we do more bootstrapping to determine which tiles are potentially
        - Ideas: https://github.com/Confirm-Solutions/confirmasaurus/issues/83
  
## Backends for Adagrid

By default we run locally on the launching machine. But, we also have tools for
launching a job on Modal and AWS. 
- `LocalBackend()` is the default
- `ModalBackend(n_zones=..., n_workers=..., gpu=...)` will launch on Modal.

## Databases for Adagrid

Adagrid depends heavily on a database backend. Why do we use the database?
- To store data that is not in memory. We have at most one packet in memory at
  a time. The database can massively reduces memory usage in active workers.
- To coordinate between distributed workers - a communications tool!
- Long-term storage of results in an accessible and stable format. It’s a nice
  extra feature that this gives us a backup!
  
Different database backends:

- Pandas. Not a real database! But it acts like one, for the sake of the code.
  This is used exclusively for testing.
- DuckDB: An "embedded database" (no need to run a server, just `import duckdb`
  and do your thing). Similar to SQLite. Optionally backed by a file but
  defaults to being only in-memory. Can't handle distributed computing. 
- Clickhouse (via Clickhouse Cloud, which bills through AWS). In a lot of ways,
  like Duckdb, but it runs like a server, and you can connect to it via
  hostname and port, so we can have a single server that then servers a lot of
  workers. We use Clickhouse for distributed jobs. 

### Database tables
- `tiles`: The `tiles` table contains all the tiles that have been created.
  These tiles may not necessarily have been simulated yet.
    - If you launch a run with 1k tiles, there will be 1k rows in the `tiles`
      table. Initially, there will be 0 rows in the `results` and `done`
      tables.
- `results`: Simulation results plus all columns from `tiles` are duplicated.
  For example, in calibration, this will contain lambda* values for each tiles.
  We duplicate columns in `tiles` so that after a run is complete, the
  `results` table can be used as a single source of information about what
  happened in a run.
- `done`: After a tile has been refined or deepened, information about the
  event is described in the `done` table. Presence in the `done` table
  indicates that the tile is no longer "active".
- `config`: Columns for arguments to the `ada_validate` and `ada_calibrate`
  functions plus other system configuration information. A row is added for
  each worker. Each row includes details of the hardware it’s running on,
  packages are installed, what versions of packages, etc. This is meant to
  track the setup, we don’t actually use it for the run, it’s for
  reproducibility.
- `null_hypos`: The null hypotheses.
- `reports`: not read from during the run. A record, broadly, of what was done
  during each step of of adagrid, for each worker. So, broadly, each time a
  worker does a Thing (new step, process a packet, coordination), it inserts a
  row into reports.

Why do we have 3 different ways of representing the state of tiles? So that we
can record information in an append-only fashion. This makes it very easy to
see what has happened, and it has nice robustness properties because we never
have to wonder if the row is updated or not (if it exists, it is in final
state; so the only question is whether the row is there.) So, the easy way to
ask, “has this tile been simulated yet,” is to query, “are the results of this
tile in the results table.”

### Notes:
- Every job has a `job_id` - a name. This can be manually chosen but also can
  be assigned randomly.
- Every worker that joins an adagrid job receives a sequential worker ID. 
    - worker_id=0 is reserved and should not be used
    - worker_id=1 is the default used for non-adagrid, non-distributed jobs
      (e.g. ip.validate, ip.calibrate). It is also used for the leader in
      distributed adagrid jobs.
    - worker_id>=2 are workers that join the distributed adagrid job.