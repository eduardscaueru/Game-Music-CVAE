from style_rank import get_similarity_matrix, get_feature_names


def similarity_matrix(rank_set, style_set, feature_names=None):
    if feature_names is None:
        feature_names = []
    return get_similarity_matrix(rank_set, style_set, feature_names=feature_names, return_paths_and_labels=True)


def get_all_feature_names():
    return get_feature_names("ALL")


if __name__ == "__main__":
    features = get_feature_names()
    print(similarity_matrix(["/home/ediuso/Documents/Licenta/Game-Music-CVAE/data/adventure/myst/MystTheme.mid"],
                                ["/home/ediuso/Documents/Licenta/Game-Music-CVAE/data/adventure/blade_runner/bladerun.mid",
                                 "/home/ediuso/Documents/Licenta/Game-Music-CVAE/data/adventure/myst/mystflight.mid",
                                 "/home/ediuso/Documents/Licenta/Game-Music-CVAE/data/arcade/blox/Bach - Bourree.mid"]))
