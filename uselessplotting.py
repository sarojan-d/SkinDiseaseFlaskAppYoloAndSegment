import matplotlib.pyplot as plt

# Function to read results.txt and parse data
def parse_results(file_path):
    data = {
        'epoch': [], 'box_loss': [], 'obj_loss': [], 'cls_loss': [], 'total_loss': [],
        'val_box_loss': [], 'val_obj_loss': [], 'val_cls_loss': [],
        'map_0.5': [], 'map_0.5_0.95': []
    }

    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():  # Skip empty lines
                parts = line.split()
                if len(parts) >= 15:  # Ensure sufficient columns are present
                    data['epoch'].append(int(parts[0].split('/')[0]))
                    data['box_loss'].append(float(parts[2]))
                    data['obj_loss'].append(float(parts[3]))
                    data['cls_loss'].append(float(parts[4]))
                    data['total_loss'].append(float(parts[5]))
                    data['map_0.5'].append(float(parts[10]))
                    data['map_0.5_0.95'].append(float(parts[11]))

    return data

# Function to plot metrics
def plot_metrics(data):
    epochs = data['epoch']

    # Plot losses
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(epochs, data['box_loss'], label='Box Loss', color='blue')
    plt.plot(epochs, data['obj_loss'], label='Objectness Loss', color='green')
    plt.plot(epochs, data['cls_loss'], label='Class Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Losses')
    plt.legend()

    # Plot mAPs
    plt.subplot(2, 1, 2)
    plt.plot(epochs, data['map_0.5'], label='mAP@0.5', color='purple')
    plt.plot(epochs, data['map_0.5_0.95'], label='mAP@0.5:0.95', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('mAP')
    plt.title('Validation mAPs')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Usage
file_path = 'runs\train'  # Update this to the correct path
data = parse_results(file_path)
plot_metrics(data)
