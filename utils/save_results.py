import os
import numpy as np


def save_results(history, save_path, n_class, batch_size, epochs):
    avg_acc = np.mean(history)
    std_acc = np.std(history)
    title = 'classes   \tdropout   \tbatch_size\tepochs    \n'
    parameters = f'{n_class:<10}\t{batch_size:<10}\t{epochs:<10}\n'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    filename = os.path.join(save_path, f'{n_class}.txt')
    sub_avg = np.round(np.mean(history, axis=1), decimals=4)  # 计算行平均值

    # Check if file already exists
    count = 1
    while os.path.exists(filename):
        filename = os.path.join(save_path, f'{n_class}_{count}.txt')
        count += 1

    # Write to file
    with open(filename, 'w') as file:
        # Save transposed numpy array to file
        file.write('All Results\n')
        file.write('sub1\tsub2\tsub3\tsub4\tsub5\tsub6\tsub7\tsub8\tsub9\tsub10\n')
        np.savetxt(file, np.round(history.T, decimals=4), delimiter='\t', fmt='%0.4f')
        file.write('\n')
        file.write(f'Sub Average:\n')
        np.savetxt(file, [sub_avg], delimiter='\t', fmt='%0.4f')  # 将行平均值保存到文件中
        file.write('\n')
        file.write(f'Parameters:\n')
        file.write(title)
        file.write(parameters)
        file.write('\n')
        file.write(f'Average Accuracy:\t{avg_acc:.4f}±{std_acc:.4f}\n')


if __name__ == '__main__':
    test_history = np.random.rand(10, 10)
    test_history = np.array(test_history)
    save_results(test_history, './outputs/72-Exemplar', 72, 64, 70)

