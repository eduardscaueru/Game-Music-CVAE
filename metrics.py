from style_rank import get_similarity_matrix, get_feature_names


def similarity_matrix(rank_set, style_set, feature_names=None):
    if feature_names is None:
        feature_names = []
    return get_similarity_matrix(rank_set, style_set, feature_names=feature_names, return_paths_and_labels=True)


def get_all_feature_names():
    return get_feature_names("ALL")


if __name__ == "__main__":
    features = get_feature_names()
    print(similarity_matrix(["/home/ediuso/Documents/Licenta/Game-Music-CVAE/out/generated_doom+burning_monkey_changelog_5.mid",
                             "/home/ediuso/Documents/Licenta/Game-Music-CVAE/out/generated_doom+burning_monkey_changelog_6.mid"],
                                ["/home/ediuso/Documents/Licenta/Game-Music-CVAE/data/action/doom/02 - At Doom's Gate (E1M1).mid",
                                 "/home/ediuso/Documents/Licenta/Game-Music-CVAE/data/arcade/burning_monkey/FilthyTouch.mid"]))
