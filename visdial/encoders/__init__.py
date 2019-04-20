from visdial.encoders.lf import LateFusionEncoder
from visdial.encoders.hr import HierarchicalRecurrentEncoder


def Encoder(model_config, *args):
    name_enc_map = {"lf": LateFusionEncoder, "hr": HierarchicalRecurrentEncoder}
    return name_enc_map[model_config["encoder"]](model_config, *args)
