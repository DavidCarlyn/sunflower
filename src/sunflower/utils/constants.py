from enum import StrEnum


class DatasetNames(StrEnum):
    CUB = "CUB"
    NABIRDS = "NABIRDS"
    IMAGENET = "IMAGENET"
    FLOWERS = "FLOWERS"
    MNIST = "MNIST"
    CELEBA = "CELEBA"
    CELEBA_HQ = "CELEBA_HQ"
    ZEBRA_SORREL = "ZEBRA_SORREL"
    CHEETAH_COUGAR = "CHEETAH_COUGAR"
    EGYPTIAN_PERSIAN = "EGYPTIAN_PERSIAN"


class SDVAEPretrainedModels(StrEnum):
    SD15 = "runwayml/stable-diffusion-v1-5"
    OSTRIS = "ostris/vae-kl-f8-d16"
