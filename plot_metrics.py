import matplotlib.pyplot as plt

# Hardcoded metrics directly from your successful training terminal run
epochs = [1, 2, 3, 4, 5]
training_loss = [1.0135, 0.5175, 0.4122, 0.3618, 0.3249]
validation_accuracy = [85.30, 88.40, 89.10, 90.00, 91.30]

# Set up a clean, professional dual-plot figure
plt.figure(figsize=(12, 5))

# 1. Plot Loss Curve
plt.subplot(1, 2, 1)
plt.plot(epochs, training_loss, marker='o', color='#e74c3c', linewidth=2, label='Training Loss')
plt.title('Model Convergence (Cross-Entropy Loss)', fontsize=12, fontweight='bold', pad=10)
plt.xlabel('Epoch', fontsize=10)
plt.ylabel('Loss Value', fontsize=10)
plt.xticks(epochs)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

# 2. Plot Accuracy Curve
plt.subplot(1, 2, 2)
plt.plot(epochs, validation_accuracy, marker='s', color='#2ecc71', linewidth=2, label='Validation Accuracy')
plt.axhline(y=91.30, color='#2c3e50', linestyle=':', alpha=0.8, label='Final Milestone (91.30%)')
plt.title('Pathogen Classification Accuracy', fontsize=12, fontweight='bold', pad=10)
plt.xlabel('Epoch', fontsize=10)
plt.ylabel('Accuracy (%)', fontsize=10)
plt.xticks(epochs)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

plt.tight_layout()

# Save the plot directly into your project folder
output_filename = "tomato_training_performance.png"
plt.savefig(output_filename, dpi=300)
print(f"📊 Success! High-resolution plot saved as '{output_filename}' in your project folder.")