import numpy as np
from eval_utils import *
from sklearn.metrics.pairwise import cosine_similarity
from config.config import config as cfg
from sklearn import preprocessing
import cv2
import os


def get_embeddings(model_path, datasets):
    device_id = 0
    backbone, device = load_model(path=model_path, embedding_size=cfg.embedding_size, device_id=device_id)
    embs = []
    for ids, ds in enumerate(datasets):
        print(f"Extracting {ds['name']}")

        with open(ds['pairs_file'], "r") as cmpfile:
            cmps = cmpfile.readlines()

        a, b = [], []
        for c in cmps:
            a1, b1 = c.strip().split(";")
            a.append(a1)
            b.append(b1)

        images = torch.zeros((len(a) + len(b), 3, 112, 112))
        i = 0
        stride = len(a)
        for a_p, b_p in zip(a, b):
            try:
                images[i] = torch.from_numpy(
                    np.transpose(cv2.cvtColor(cv2.imread(os.path.join(ds['root'], a_p)), cv2.COLOR_BGR2RGB),
                                 axes=(2, 0, 1)))
                images[i + stride] = torch.from_numpy(
                    np.transpose(cv2.cvtColor(cv2.imread(os.path.join(ds['root'], b_p)), cv2.COLOR_BGR2RGB),
                                 axes=(2, 0, 1)))
                i += 1
            except:
                print(a_p, b_p)
        batch_size = 10
        ba = 0
        embeddings = np.zeros((images.shape[0], cfg.embedding_size))
        while ba < images.shape[0]:
            bb = min(ba + batch_size, images.shape[0])
            count = bb - ba
            _data = images[bb - batch_size: bb]
            _data = ((_data / 255) - 0.5) / 0.5
            net_out: torch.Tensor = backbone(_data)
            _embeddings = net_out.detach().cpu().numpy()

            embeddings[ba:bb, :] = _embeddings[(batch_size - count):, :]
            ba = bb
        embeddings = preprocessing.normalize(embeddings)
        embs.append(embeddings)
    return embs


def evaluate():
    models = {
        "baseline": cfg.base_model_path,
        "proposed": cfg.mixed_model_path
    }

    datasets = [cfg.AgeDB, cfg.BUPT, cfg.ROF, cfg.CFPFP]
    for key, method in models.items():
        _embs = get_embeddings(method, datasets)

        if not os.path.exists("wacv_results"):
            os.makedirs("wacv_results")

        for _, r_np in enumerate(_embs):
            s_tot = r_np.shape[0]
            e1 = r_np[:int(s_tot / 2)]
            e2 = r_np[int(s_tot / 2):]
            sims = np.diag(cosine_similarity(e1, e2))
            is_same = np.zeros(shape=(sims.shape[0]), dtype=np.int32)
            is_same[sims > cfg.thresholds[key]] = 1

            if not os.path.exists(os.path.join("wacv_results", datasets[_]['name'])):
                os.makedirs(os.path.join("wacv_results", datasets[_]['name']))
            with open(os.path.join("wacv_results", datasets[_]['name'], f"{key}.txt"), "w") as res_file:
                for s, d in zip(sims, is_same):
                    res_file.write(f"{s},{d}\n")
            res_file.close()


if __name__ == '__main__':
    evaluate()
