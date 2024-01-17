import argparse
import multiprocessing
import os
import warnings

import numpy as np
from img2vec_pytorch import Img2Vec
from PIL import Image


class img2vec:
    def __init__(self, cuda: bool):
        self.i2v = Img2Vec(cuda=cuda)

    def get_vec(self, img_path: str) -> list:
        img = Image.open(img_path)
        return self.i2v.get_vec(img)

    def calc_dist(self, img1: str, img2: str):
        vec1 = self.get_vec(img1)
        vec2 = self.get_vec(img2)
        cosine_distance = 1 - np.dot(vec1, vec2) / (
            np.linalg.norm(vec1) * np.linalg.norm(vec2)
        )
        return cosine_distance


def is_valid_directory(path: str) -> bool:
    """Проверка, что путь существует и является директорией

    Args:
        path (str): относительный путь

    Returns:
        bool: True, если существует
    """
    if os.path.exists(path) and os.path.isdir(path):
        return True
    else:
        return False


def process_item(img1):
    match = []
    for img2 in range(img1 + 1, len(image_files)):
        dist = 1 - np.dot(vec_from_files[img1], vec_from_files[img2]) / (
            np.linalg.norm(vec_from_files[img1]) * np.linalg.norm(vec_from_files[img2])
        )

        if dist < th:
            match.append(image_files[img1])
            match.append(image_files[img2])
    return match


def main():
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(
        description="Вывод названия всех одинаковых изображений"
    )
    parser.add_argument("path", help="Относительный путь к директории")
    parser.add_argument("--cuda", action="store_true", help="Использовать GPU")
    parser.add_argument(
        "--delete", action="store_true", help="Удалить найденные изображения"
    )
    parser.add_argument(
        "--th", type=float, default=0.1, help="threshold для сравнения векторов"
    )
    args = parser.parse_args()

    if args.cuda:
        print("Используем GPU")
    else:
        print("Используем CPU")

    if not is_valid_directory(args.path):
        print(args.path)
        print("Указанный путь не является валидной директорией")
        return

    global image_files
    global img_finder
    global th
    global vec_from_files
    th = args.th
    img_finder = img2vec(bool(args.cuda))
    image_files = [
        os.path.join(args.path, file)
        for file in os.listdir(args.path)
        if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ]
    vec_from_files = [img_finder.get_vec(x) for x in image_files]

    pool = multiprocessing.Pool()
    r = range(len(image_files) - 1)
    results = pool.map(process_item, r)
    pool.terminate()
    flat_list = [item for sublist in results if sublist for item in sublist]
    print(
        f"Всего найдено {len(flat_list) / 2} дубликатов ({round(len(flat_list) / len(image_files), 2) * 100 / 2}%)"
    )
    for file in range(0, len(flat_list), 2):
        print(
            str(flat_list[file])
            + "\n"
            + str(flat_list[file + 1])
            + "\n------------------"
        )
    if args.delete:
        for file in flat_list[1::2]:
            if os.path.exists(file):
                os.remove(file)


if __name__ == "__main__":
    main()
