from verifier_bottleneck.manifests import build_all_manifests

if __name__ == "__main__":
    build_all_manifests(
        output_dir="data/manifests",
        seed=42,
        n_warmup_l1=100,
        n_source_l2=500,
        n_heldout_l2=200,
        n_transfer_l3=100,
    )

    print("Manifests written to data/manifests/")