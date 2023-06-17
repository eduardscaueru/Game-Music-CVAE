from style_rank import get_similarity_matrix, get_feature_names


def similarity_matrix(rank_set, style_set, feature_names=None):
    if feature_names is None:
        feature_names = []
    return get_similarity_matrix(rank_set, style_set, feature_names=feature_names, return_paths_and_labels=True)


def get_all_feature_names():
    return get_feature_names("ALL")


if __name__ == "__main__":
    features = get_feature_names()
    print(similarity_matrix(["/home/ediuso/Documents/Licenta/Game-Music-CVAE/out/generated_blox_changelog_10_epoch_3600_greedy.mid"],
                                ["/home/ediuso/Documents/Licenta/Game-Music-CVAE/data/arcade/blox/Bach - Bourree.mid",
                                 "/home/ediuso/Documents/Licenta/Game-Music-CVAE/data/arcade/blox/Mussorgsky - Promenade(blox).mid"]))
    # print(similarity_matrix(
    #     ["/home/ediuso/Documents/Licenta/Game-Music-CVAE/out/generated_kung_fu+burning_monkey_changelog_10_epoch_4000_random.mid"],
    #     ["/home/ediuso/Documents/Licenta/Game-Music-CVAE/data/action/kung_fu/T_kungfu.mid",
    #      "/home/ediuso/Documents/Licenta/Game-Music-CVAE/data/arcade/burning_monkey/FilthyTouch.mid"]))
