from sklearn.model_selection import KFold
import pickle

from dataset import EEGDataset

num_class = 6
data = 'EEG72'
dataPath1 = f'H:\\EEG\\EEGDATA\\{data}'
dataPath2 = f'H:\\EEG\\EEGDATA\\{data}-CWT'

# 创建KFold对象
k = 10
k_fold = KFold(n_splits=k, shuffle=True, random_state=42)

def create_kfold_indices():
    all_indices = []
    for i in range(0, 10):
        # 数据集
        dataset = EEGDataset(file_path1=dataPath1 + f'\\S{i + 1}.mat', file_path2=dataPath2 + f'\\S{i + 1}_CWT.npy',
                             num_class=num_class)

        indices = [(train_index, test_index) for train_index, test_index in k_fold.split(dataset)]

        all_indices.append(indices)
        print(f'Sub {i + 1} indices created.')

    # 保存所有的索引到同一个文件中
    with open(f'kfold_indices_{num_class}.pkl', 'wb') as f:
        pickle.dump(all_indices, f)
    print(f'All indices created and saved to kfold_indices_{num_class}.pkl')


if __name__ == '__main__':
    create_kfold_indices()
