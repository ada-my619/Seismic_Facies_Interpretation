import matplotlib.pyplot as plt

def viz_dataset_by_idx(dataset, idx):
    x, y, d = dataset[idx]
    print(d)
    fig = plt.figure(figsize=(12, 4))

    fig.suptitle(f'Slice index: {idx} Direction: {d}', fontsize=16)

    ax1 = plt.subplot(1, 2, 1)
    ax1.imshow(x.permute(2, 1, 0).numpy(), cmap='seismic', aspect='auto')
    ax1.set_title('Seismic Section')

    ax2 = plt.subplot(1, 2, 2)
    ax2.imshow(y.permute(1, 0).numpy(), cmap='seismic', aspect='auto')
    ax2.set_title('Facies Interpretation')

    plt.show()