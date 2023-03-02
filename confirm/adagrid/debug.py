def get_logs(job_id, query="select * from logs order by t"):
    """
    A debugging and observation tool for getting the logs from a job.

    Args:
        job_id: The job id.
        query: The select query. This can be used to filter by worker_id or
            time or message type or anything else! Defaults to 'select * from logs
            order by t'.

    Returns:
        (str, pd.DataFrame, Clickhouse)
        The first element is a string containing the logs, formatted similarly
            to the realtime stdout loggign
        The second element is a pandas DataFrame containing the raw log data.
        The third element is the Clickhouse database connection. This can be
            used for further investigation.

    """
    import confirm.cloud.clickhouse as ch

    db = ch.Clickhouse.connect(job_id)
    log_df = ch._query_df(db.client, query)
    log_df["t_str"] = log_df["t"].apply(lambda x: x.strftime("%Y-%m-%d %H:%M:%S.%f"))
    return reconstruct_messages(log_df), log_df


def reconstruct_messages(log_df):
    return "\n".join(
        [
            f"[worker_id={row['worker_id']}] {row['t_str']}"
            f" - {row['name']} - {row['levelname']}\n"
            f"{row['message']}"
            for _, row in log_df.iterrows()
        ]
    )


# TODO: let's make this work for arbitrary slices?
def summarize_validate(g, rej_df, plot_dims=(0, 1)):
    import matplotlib.pyplot as plt

    d1, d2 = plot_dims
    plt.figure(figsize=(10, 10), constrained_layout=True)
    plt.subplot(2, 2, 1)
    plt.title("Overall bound")
    plt.scatter(g.df[f"theta{d1}"], g.df[f"theta{d2}"], c=rej_df["tie_bound"])
    plt.xlabel(r"$\theta_0$")
    plt.ylabel(r"$\theta_1$")
    plt.colorbar(label=r"$U$")

    plt.subplot(2, 2, 2)
    plt.title("TIE at simulation points")
    plt.scatter(g.df[f"theta{d1}"], g.df[f"theta{d2}"], c=rej_df["tie_est"])
    plt.xlabel(r"$\theta_0$")
    plt.ylabel(r"$\theta_1$")
    plt.colorbar(label=r"$\hat{f}$")

    plt.subplot(2, 2, 3)
    plt.title("CSE cost")
    plt.scatter(
        g.df[f"theta{d1}"],
        g.df[f"theta{d2}"],
        c=rej_df["tie_bound"] - rej_df["tie_cp_bound"],
    )
    plt.xlabel(r"$\theta_0$")
    plt.ylabel(r"$\theta_1$")
    plt.colorbar(label=r"cost")

    plt.subplot(2, 2, 4)
    plt.title("Clopper-Pearson cost")
    plt.scatter(
        g.df[f"theta{d1}"],
        g.df[f"theta{d2}"],
        c=rej_df["tie_cp_bound"] - rej_df["tie_est"],
    )
    plt.xlabel(r"$\theta_0$")
    plt.ylabel(r"$\theta_1$")
    plt.colorbar(label=r"cost")
