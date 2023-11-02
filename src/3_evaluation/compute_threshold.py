import numpy as np
from eval_utils import *
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from config.config import config as cfg
import sklearn


def get_threshold(model_path, real_data, synthetic_data):
    device_id = 0
    backbone, device = load_model(path=model_path, embedding_size=cfg.embedding_size, device_id=device_id)
    dataset, dataloader, batch_size = get_dataloader(real_path=real_data, synth_path=synthetic_data,
                                                     local_rank=device_id)
    is_synt = np.asarray(dataset.is_synth).reshape(-1, 1)
    embeddings = np.zeros(shape=(len(dataset), 512))
    labels = np.zeros(shape=(len(dataset), 1))
    synth = np.zeros(shape=(len(dataset), 1))
    start = 0
    with torch.no_grad():
        for _, (idx, img, label) in enumerate(dataloader):
            img = img.cuda(1, non_blocking=True)
            features = backbone(img)
            embeddings[start: start + batch_size, :] = features.cpu().numpy()
            synth[start: start + batch_size] = is_synt[idx]
            labels[start: start + batch_size] = label.cpu().numpy().reshape(-1, 1)
            start = start + batch_size
    data = np.hstack([embeddings, labels, synth])

    data = data[:, :-1]
    data[:, :-1] = sklearn.preprocessing.normalize(data[:, :-1])
    identities = np.unique(data[:, -1])
    id_idx = np.asarray(list(range(len(identities))))
    pos, neg = [], []

    print("Computing positive scores...")
    with tqdm(total=len(identities)) as pbar:
        for i, id in enumerate(identities):
            pos_mask = data[:, -1] == id
            pos_sim = cosine_similarity(data[pos_mask, :-1]).flatten()
            no_m = np.delete(pos_sim, np.where(pos_sim == 1))
            pos.append(np.mean(no_m))
            pbar.update(1)

    print("Computing negative scores...")
    size = 10000
    _idx = np.random.choice(data.shape[0], size, replace=False)
    e1_idx = _idx[:size // 2]
    e2_idx = _idx[size // 2:]
    similarities = cosine_similarity(data[e1_idx, :-1], data[e2_idx, :-1])

    match_matrix = data[e1_idx, -1][:, None] - data[e2_idx, -1][None, :]
    match_mask = match_matrix == 0

    negatives = similarities[~match_mask].flatten()
    thresholds = np.arange(-1, 1, 0.001)
    fnmrs, fmrs, best_t = np.inf, np.inf, 0

    tot_pos = pos_sim.shape[0]
    tot_neg = negatives.shape[0]

    print("Computing optimal threshold...")
    with tqdm(total=len(thresholds)) as pbar:
        for t in thresholds:
            fnmrs_t = (pos_sim < t).sum() / tot_pos
            fmrs_t = (negatives > t).sum() / tot_neg
            if fnmrs_t + fmrs_t <= fnmrs + fmrs:
                fnmrs = fnmrs_t
                fmrs = fmrs_t
                best_t = t
            pbar.update(1)

    return best_t


def compute():
    baseline = {
        "authentic_root": cfg.authentic_root,
        "synthetic_root": None,
        "model": cfg.base_model_path
    }

    proposed = {
        "authentic_root": cfg.authentic_root,
        "synthetic_root": cfg.synthetic_root,
        "model": cfg.mixed_model_path
    }
    best_ts = []
    for method in [baseline, proposed]:
        _t = get_threshold(method["model"], method["authentic_root"], method["synthetic_root"])
        best_ts.append(_t)

    print(f"OPTIMAL THRESHOLDS \n"
          f"baseline: {best_ts[0]} - proposed: {best_ts[1]}")


if __name__ == '__main__':
    compute()
