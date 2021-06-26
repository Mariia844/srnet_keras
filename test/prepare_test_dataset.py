import os

# from main import train_model
SYMLINK_PATH = 'D:/mary_study/test_2k'
TRAIN_SOURCES = ['D:/mary_study/ALASKA_v2_TIFF_512_GrayScale_PGM', 'D:/mary_study/stego/S-UNIWARD/50']
TRAIN_COUNT = 2000
VALIDATION_COUNT = 1000

import model_config

if __name__ == '__main__':
    # os.makedirs(SYMLINK_PATH)
    # train_path = os.path.join(SYMLINK_PATH, 'train')
    # validation_path = os.path.join(SYMLINK_PATH, 'validation')
    # os.makedirs(train_path)
    # os.makedirs(validation_path)
    # for i in range(len(TRAIN_SOURCES)):
    #     train_class_path = os.path.join(train_path, str(i))
    #     validation_class_path = os.path.join(validation_path, str(i))
    #     all_filenames = os.listdir(TRAIN_SOURCES[i])
    #     os.makedirs(train_class_path)
    #     os.makedirs(validation_class_path)
    #     for j in range(TRAIN_COUNT):
    #         filename = all_filenames[j]
    #         source_path = os.path.join(TRAIN_SOURCES[i], filename)
    #         destination_path = os.path.join(train_class_path, filename)
    #         os.symlink(source_path, destination_path)
    #     for k in range(TRAIN_COUNT, TRAIN_COUNT + VALIDATION_COUNT):
    #         filename = all_filenames[k]
    #         source_path = os.path.join(TRAIN_SOURCES[i], filename)
    #         destination_path = os.path.join(validation_class_path, filename)
    #         os.symlink(source_path, destination_path)
        
    os.symlink(model_config.COVER_PATH, os.path.join(model_config.SYMLINK_PATH, 'cover'))
    os.symlink(os.path.join(model_config.PATH, model_config.SYMLINK_ALGORYTHM), os.path.join(model_config.SYMLINK_PATH, 'stego'))