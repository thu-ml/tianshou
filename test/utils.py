from tianshou.data.collector import CollectStats


def print_final_stats(collect_stats: CollectStats) -> None:
    if collect_stats.returns_stat is not None and collect_stats.lens_stat is not None:
        print(
            f"Final reward: {collect_stats.returns_stat.mean}, length: {collect_stats.lens_stat.mean}",
        )
    else:
        print("Final stats rollout not available.")
