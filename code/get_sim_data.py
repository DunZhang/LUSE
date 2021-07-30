import os
import random

path_info = {
    "adat": ("adat_train.txt", "adat_dev.txt"),
    "atec": ("atec_train.txt", "atec_dev.txt"),
    "ccks": ("ccsk_train.txt", "ccks_dev.txt"),
    "lcqmc": ("lcqmc_train.txt", "lcqmc_dev.txt", "lcqmc_test.txt"),
}


def get_sim_data(data_dir, save_path, max_count=30000):
    sens = []
    for names in path_info.values():
        domain_sens = []
        for name in names:
            with open(os.path.join(data_dir, name), "r", encoding="utf8") as fr:
                for line in fr:
                    ss = line.strip().split("\t")
                    sen1, sen2 = ss[:2]
                    domain_sens.append(sen1.strip())
                    domain_sens.append(sen2.strip())
        domain_sens = list(set(domain_sens))
        random.shuffle(domain_sens)
        if max_count is not None:
            domain_sens = domain_sens[:max_count]
        sens.extend(domain_sens)
    random.shuffle(sens)
    with open(save_path, "w", encoding="utf8") as fw:
        fw.writelines([i + "\n" for i in sens])


if __name__ == "__main__":
    get_sim_data(data_dir=r"H:\我的坚果云\文本相似度数据集\SimilarityData",
                 save_path=r"G:\Codes\LUSE\sim_data.txt")
