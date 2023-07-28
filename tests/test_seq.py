import numpy as np


def test_data(actions, lengths, exercises, targets, metadata, indices):
    # Everything has same length
    all_length_values = set(map(len, [actions, lengths, exercises, targets, indices]))
    assert len(all_length_values) == 1

    n_students = list(all_length_values)[0]
    n_samples = sum(lengths) + n_students

    some_length = len(actions[0])
    max_length = max(lengths)
    assert some_length <= max_length

    assert targets.flatten().max() == 1

    n_metadata_rows = metadata.shape[0]

    assert n_metadata_rows == n_samples

    max_indice = max(max(indice_list) for indice_list in indices)
    assert max_indice < n_metadata_rows

    assert max_indice == n_metadata_rows - 1  # Pas garanti à 100 %

    # Checksum
    # print(actions.sum(), lengths.sum(), exercises.sum(), targets.sum(), metadata.sum(), indices.sum())
    # print('first line', metadata[:1], metadata[:1].sum())
    # Fraction 132009 10184 101840 5438 331248.0 54586240 (10720 x 44)
    # Clean Fraction 132009 10184 101840 5438 40736.0 54586240 (10720 x 44)
