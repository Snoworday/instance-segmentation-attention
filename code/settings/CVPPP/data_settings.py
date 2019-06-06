import os

class DataSettings(object):

    def __init__(self):

        self.BASE_PATH = os.path.abspath(
            os.path.join(
                __file__,
                os.path.pardir,
                os.path.pardir,
                os.path.pardir,
                os.path.pardir))

        self.CLASS_WEIGHTS = None

        self.MAX_N_OBJECTS = 32

        self.N_CLASSES = 1 + 1
