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
