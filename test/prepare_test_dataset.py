import os
import model_config
if __name__ == '__main__':
    os.symlink(model_config.COVER_PATH, os.path.join(model_config.SYMLINK_PATH, 'cover'))
    os.symlink(os.path.join(model_config.PATH, model_config.SYMLINK_ALGORYTHM), os.path.join(model_config.SYMLINK_PATH, 'stego'))