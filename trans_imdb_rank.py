import json
import numpy as np
import torch
import os
for filename in ['npy_folder/imdb_bert_512_score.npy', 'npy_folder/imdb_distilbert_512_score.npy']:
    if not os.path.exists(filename):
        continue
    imdb_scores = np.load(filename)
    features_and_dataset = torch.load('imdb/cached_train_512_lower')
    dataset = features_and_dataset["dataset"]
    all_input_mask = dataset.tensors[1]
    L = all_input_mask.sum(dim=-1)

    assert(len(L)==len(imdb_scores))

    rank = []
    for i in range(len(L)):
        imdb_score = imdb_scores[i]
        l = int(L[i])

        gs = []
        for widx in range(len(imdb_score)):
            tmp = imdb_score[widx]

            order = np.argsort(-tmp[:l])

            guide = np.zeros(len(tmp), dtype=np.float32)
            for idx, x in enumerate(order):
                guide[x] = 1-idx/l
            gs.append(guide)
        gs = np.array(gs)

        rank.append(gs)

    rank = np.array(rank)
    o_filename = filename.replace('score' , 'rank')
    np.save(o_filename, rank)

