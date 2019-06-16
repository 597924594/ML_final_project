from tqdm import tqdm

# pbar = tqdm(["a", "b", "c", "d"])
# for char in pbar:
#     # pbar.set_description("Processing %s" % char)
#     tqdm.write("Processing %s" % char)

with tqdm(total=100) as pbar:
    for i in range(10):
        pbar.update(10)