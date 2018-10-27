"""
registry for model constructors, used by ModelManager to lookup model constructors
"""
from libs.models.BagOfWords import BagOfWords
from libs.models.NLIRNN import NLIRNN

# todo make the registry dynamic based on the file name

reg = {
    'BagOfWords': BagOfWords,
    'NLIRNN': NLIRNN
}


class ModelRegister:
    def __init__(self):
        for k in reg.keys():
            setattr(self, k, k)

    @property
    def model_list(self):
        return '\n=== Models Available ===\n{}\n========================'.format(
            '\n'.join(self.__dict__.keys()))


modelRegister = ModelRegister()
