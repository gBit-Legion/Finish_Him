from Services.unpacker import *
import os


def extractor(path):
    UP = UnPacker(path)

    if os.path.splitext(path)[1].lower() == ".rar":
        UP.rar_unpacker()
    elif os.path.splitext(path)[1].lower() == ".zip":
        UP.zipper()
    elif os.path.splitext(path)[1].lower() == ".tar":
        UP.tar_unpacker()
