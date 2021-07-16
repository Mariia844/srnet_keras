# PATH = "/kaggle/input/alaska2-image-steganalysis/"
PATH = "D:\mary_study\stego\S-UNIWARD".replace("\\", "/")
# ALGORITHMS = [str(_) for _ in range(10, 60, 10)] # 10 - 50 with step 10
COVER_PATH = "D:\mary_study\ALASKA_v2_TIFF_512_GrayScale_PGM"
ALGORITHMS = ['50']
IMG_SIZE = 512
RGB = False
CPU_COUNT = 6

# Model training config
EPOCHS = 5
BATCH_SIZE = 16

# Symlink config
SYMLINK_PATH = "D:/mary_study/test_2k"
SYMLINK_ALGORYTHM = '50'