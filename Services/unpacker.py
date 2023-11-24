import zipfile
import tarfile
import rarfile
import os


class UnPacker:
    def __init__(self, path_to_file):
        # Укажи путь к ZIP-файлу
        self.path = path_to_file
        if not os.path.exists("./np_array_file"):
            os.makedirs("./np_array_file")

    def zipper(self):
        print(1)
        with zipfile.ZipFile(self.path, 'r') as zip_ref:
            # Извлекаем все файлы в текущую директорию
            zip_ref.extractall("./np_array_file")

    def tar_unpacker(self):

        with tarfile.TarFile(self.path, 'r') as ref:
            # Извлекаем все файлы в текущую директорию
            ref.extractall("./np_array_file")

    def rar_unpacker(self):

        with rarfile.RarFile(self.path, 'r') as ref:
            # Извлекаем все файлы в текущую директорию
            ref.extractall("./np_array_file")
