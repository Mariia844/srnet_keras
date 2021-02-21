# PATH = "/kaggle/input/alaska2-image-steganalysis/"
PATH = "./MIRFLICKR_part1_MG_1times"
# ALGORITHMS = [str(_) for _ in range(10, 60, 10)] # 10 - 50 with step 10
ALGORITHMS = ['50']
IMG_SIZE = 256
RGB = False
CPU_COUNT = 6

# Model training config
EPOCHS = 5
BATCH_SIZE = 16
