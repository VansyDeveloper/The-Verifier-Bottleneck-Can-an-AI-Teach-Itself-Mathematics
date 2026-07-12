from verifier_bottleneck.manifests import validate_all_manifests

if __name__ == "__main__":
    validate_all_manifests("data/manifests")
    print("All manifests are valid.")