"""
registry for loader constructors, used by ModelManager to lookup data loaders
"""
from libs.data_loaders.ImdbLoader import ImdbLoader
from libs.data_loaders.SNLILoader import SNLILoader

# todo make the registry dynamic based on the file name

reg = {
    'IMDB': ImdbLoader,
    'SNLI': SNLILoader
}


class LoaderRegister:
    def __init__(self):
        for k in reg.keys():
            setattr(self, k, k)

    @property
    def loader_list(self):
        return '\n=== Loaders Available ===\n{}\n========================'.format(
            '\n'.join(self.__dict__.keys()))


loaderRegister = LoaderRegister()
