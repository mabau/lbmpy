from enum import Enum, auto


class Stencil(Enum):
    D2Q9 = auto()
    D2V17 = auto()
    D2V37 = auto()
    D3Q15 = auto()
    D3Q19 = auto()
    D3Q27 = auto()


class Method(Enum):
    SRT = auto()
    TRT = auto()
    MRT = auto()
    CENTRAL_MOMENT = auto()
    MRT_RAW = auto()
    TRT_KBC_N1 = auto()
    TRT_KBC_N2 = auto()
    TRT_KBC_N3 = auto()
    TRT_KBC_N4 = auto()
    ENTROPIC_SRT = auto()
    CUMULANT = auto()
    MONOMIAL_CUMULANT = auto()


class ForceModel(Enum):
    SIMPLE = auto()
    LUO = auto()
    GUO = auto()
    BUICK = auto()
    SILVA = auto()
    EDM = auto()
    KUPERSHTOKH = auto()
    CUMULANT = auto()
    HE = auto()
    SHANCHEN = auto()
