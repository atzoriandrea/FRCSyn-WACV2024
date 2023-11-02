from easydict import EasyDict as edict

config = edict()

config.batch_size = 10
config.embedding_size = 512
# type of network to train [iresnet100 | iresnet50]
config.synthetic_root = "../datasets/FaceRecognition/Synthetic/dcface_0.5m_oversample_xid/images"
config.authentic_root = "../datasets/FaceRecognition/casia_training"
config.BUPT = {"root": "", "pairs_file": "", "name": "bupt"}
config.AgeDB = {"root": "", "pairs_file": "", "name": "agedb"}
config.ROF = {"root": "", "pairs_file": "", "name": "rof"}
config.CFPFP = {"root": "", "pairs_file": "", "name": "cfp-fp"}
config.network = "iresnet100"
config.base_model_path = ""
config.mixed_model_path = ""
config.SE = False  # SEModule
config.thresholds = {"baseline": 0.5,
                     "proposed": 0.5}

