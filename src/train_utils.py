import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm
import numpy as np

def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=0.001, weight_decay=1e-5, patience=10, save_dir='./'):
    """
    Train the model with early stopping and real-time visualization
    """
    import time
    import matplotlib.pyplot as plt
    from IPython.display import clear_output

    print("\n" + "="*80)
    print("MODEL TRAINING")
    print("="*80)

    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)

    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    # For early stopping
    best_val_loss = float('inf')
    best_val_acc = 0.0
    early_stop_counter = 0

    # Training history
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    # Function to update plots in real-time
    def update_plots():
        # Create figure for plots
        plt.figure(figsize=(15, 10))

        # Plot loss
        plt.subplot(2, 2, 1)
        plt.plot(train_losses, 'b-', label='Train Loss')
        plt.plot(val_losses, 'r-', label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)

        # Plot accuracy
        plt.subplot(2, 2, 2)
        plt.plot(train_accs, 'b-', label='Train Accuracy')
        plt.plot(val_accs, 'r-', label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        plt.grid(True)

        # Plot learning rate
        plt.subplot(2, 2, 3)
        current_lr = optimizer.param_groups[0]['lr']
        plt.bar(len(train_losses)-1, current_lr, color='g')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title(f'Current Learning Rate: {current_lr:.6f}')
        plt.grid(True)

        # Plot confusion matrix for validation set
        if len(val_preds) > 0:
            plt.subplot(2, 2, 4)
            cm = confusion_matrix(val_true, val_preds)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Validation Confusion Matrix')

        plt.tight_layout()

        # Save the current plot
        plt.savefig(os.path.join(save_dir, 'training_progress.png'))

        # Save the plot for this specific epoch
        plt.savefig(os.path.join(save_dir, f'epoch_{len(train_losses)}_progress.png'))

        # Close the plot to free memory
        plt.close()

    print("\nTraining Configuration:")
    print(f"  Number of epochs: {num_epochs}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Weight decay: {weight_decay}")
    print(f"  Early stopping patience: {patience}")
    print(f"  Batch size: {train_loader.batch_size}")
    print(f"  Training samples: {len(train_loader.dataset)}")
    print(f"  Validation samples: {len(val_loader.dataset)}")
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    print("\nStarting training...")

    for epoch in range(num_epochs):
        epoch_start_time = time.time()

        # Training
        model.train()
        train_loss = 0.0
        train_preds = []
        train_true = []

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

            # Store predictions
            _, preds = torch.max(outputs, 1)
            train_preds.extend(preds.cpu().numpy())
            train_true.extend(labels.cpu().numpy())

        train_loss = train_loss / len(train_loader.dataset)
        train_acc = accuracy_score(train_true, train_preds)

        # Validation
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_true = []

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                inputs, labels = inputs.to(device), labels.to(device)

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)

                # Store predictions
                _, preds = torch.max(outputs, 1)
                val_preds.extend(preds.cpu().numpy())
                val_true.extend(labels.cpu().numpy())

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = accuracy_score(val_true, val_preds)

        # Update learning rate
        scheduler.step(val_loss)

        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time

        # Print statistics
        print(f'Epoch {epoch+1}/{num_epochs}, Time: {epoch_time:.2f}s, LR: {optimizer.param_groups[0]["lr"]:.6f}')
        print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

        # Store history
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # Update plots
        update_plots()

        # Save detailed epoch results
        with open(os.path.join(save_dir, 'training_log.txt'), 'a') as f:
            f.write(f'Epoch {epoch+1}/{num_epochs}, Time: {epoch_time:.2f}s, LR: {optimizer.param_groups[0]["lr"]:.6f}\n')
            f.write(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}\n')
            f.write(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}\n\n')

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            early_stop_counter = 0
            # Save the best model
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
            print(f'  New best model saved! (Val Loss: {best_val_loss:.4f}, Val Acc: {best_val_acc:.4f})')
        else:
            early_stop_counter += 1
            print(f'  Early stopping counter: {early_stop_counter}/{patience}')
            if early_stop_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break

    # Load the best model
    model.load_state_dict(torch.load(os.path.join(save_dir, 'best_model.pth')))

    # Final plots
    plt.figure(figsize=(15, 6))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, 'b-', label='Train Loss')
    plt.plot(val_losses, 'r-', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, 'b-', label='Train Accuracy')
    plt.plot(val_accs, 'r-', label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'))

    # Save training history as CSV
    import pandas as pd
    history_df = pd.DataFrame({
        'epoch': range(1, len(train_losses) + 1),
        'train_loss': train_losses,
        'val_loss': val_losses,
        'train_acc': train_accs,
        'val_acc': val_accs
    })
    history_df.to_csv(os.path.join(save_dir, 'training_history.csv'), index=False)

    # Print training summary
    total_time = time.time() - start_time
    print("\nTraining Summary:")
    print(f"  Total training time: {total_time/60:.2f} minutes")
    print(f"  Best validation loss: {best_val_loss:.4f}")
    print(f"  Best validation accuracy: {best_val_acc:.4f}")
    print(f"  Total epochs: {len(train_losses)}")
    print("="*80)

    return model

def evaluate_model(model, test_loader, class_names, save_dir='./'):
    """
    Evaluate the model on the test set with detailed metrics and visualizations
    """
    import time
    import matplotlib.pyplot as plt
    from sklearn.metrics import classification_report, roc_curve, auc, precision_recall_curve, average_precision_score
    from itertools import cycle

    print("\n" + "="*80)
    print("MODEL EVALUATION")
    print("="*80)

    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    test_preds = []
    test_true = []
    test_probs = []  # Store probabilities for ROC curves

    print("Running inference on test set...")
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1)

            # Store predictions and probabilities
            _, preds = torch.max(outputs, 1)
            test_preds.extend(preds.cpu().numpy())
            test_true.extend(labels.cpu().numpy())
            test_probs.extend(probs.cpu().numpy())

    test_probs = np.array(test_probs)

    # Calculate metrics
    accuracy = accuracy_score(test_true, test_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(test_true, test_preds, average='weighted')

    print("\nOverall Metrics:")
    print(f'  Test Accuracy: {accuracy:.4f}')
    print(f'  Test Precision: {precision:.4f}')
    print(f'  Test Recall: {recall:.4f}')
    print(f'  Test F1 Score: {f1:.4f}')

    # Detailed classification report
    print("\nDetailed Classification Report:")
    class_report = classification_report(test_true, test_preds, target_names=class_names, digits=4)
    print(class_report)

    # Confusion matrix with normalization
    cm = confusion_matrix(test_true, test_preds)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Plot confusion matrices (raw counts and normalized)
    plt.figure(figsize=(20, 8))

    plt.subplot(1, 2, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix (Counts)')

    plt.subplot(1, 2, 2)
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix (Normalized)')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
    plt.close()

    # ROC curves (one-vs-rest)
    plt.figure(figsize=(12, 10))

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i, class_name in enumerate(class_names):
        fpr[i], tpr[i], _ = roc_curve(
            np.array(test_true) == i,
            test_probs[:, i]
        )
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot all ROC curves
    colors = cycle(['blue', 'red', 'green', 'orange', 'purple', 'brown'])
    for i, color in zip(range(len(class_names)), colors):
        plt.plot(
            fpr[i], tpr[i], color=color, lw=2,
            label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})'
        )

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'roc_curves.png'))
    plt.close()

    # Precision-Recall curves
    plt.figure(figsize=(12, 10))

    # For each class
    precision_dict = dict()
    recall_dict = dict()
    avg_precision = dict()

    for i, class_name in enumerate(class_names):
        precision_dict[i], recall_dict[i], _ = precision_recall_curve(
            np.array(test_true) == i,
            test_probs[:, i]
        )
        avg_precision[i] = average_precision_score(
            np.array(test_true) == i,
            test_probs[:, i]
        )

    # Plot Precision-Recall curve for each class
    colors = cycle(['blue', 'red', 'green', 'orange', 'purple', 'brown'])
    for i, color in zip(range(len(class_names)), colors):
        plt.plot(
            recall_dict[i], precision_dict[i], color=color, lw=2,
            label=f'{class_names[i]} (AP = {avg_precision[i]:.2f})'
        )

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'precision_recall_curves.png'))
    plt.close()

    # Save detailed results to a text file
    with open(os.path.join(save_dir, 'evaluation_results.txt'), 'w') as f:
        f.write("="*50 + "\n")
        f.write("MODEL EVALUATION RESULTS\n")
        f.write("="*50 + "\n\n")

        f.write("Overall Metrics:\n")
        f.write(f'Test Accuracy: {accuracy:.4f}\n')
        f.write(f'Test Precision: {precision:.4f}\n')
        f.write(f'Test Recall: {recall:.4f}\n')
        f.write(f'Test F1 Score: {f1:.4f}\n\n')

        f.write("Detailed Classification Report:\n")
        f.write(class_report + "\n\n")

        f.write("Confusion Matrix:\n")
        f.write(str(cm) + "\n\n")

        f.write("Normalized Confusion Matrix:\n")
        f.write(str(cm_norm) + "\n\n")

        f.write("ROC AUC Scores:\n")
        for i, class_name in enumerate(class_names):
            f.write(f'  {class_name}: {roc_auc[i]:.4f}\n')
        f.write("\n")

        f.write("Average Precision Scores:\n")
        for i, class_name in enumerate(class_names):
            f.write(f'  {class_name}: {avg_precision[i]:.4f}\n')
        f.write("\n")

        f.write("Evaluation completed in {:.2f} seconds\n".format(time.time() - start_time))

    # Save predictions to CSV for further analysis
    import pandas as pd
    predictions_df = pd.DataFrame({
        'true_label': [class_names[i] for i in test_true],
        'predicted_label': [class_names[i] for i in test_preds],
        'correct': np.array(test_true) == np.array(test_preds)
    })

    # Add probability columns for each class
    for i, class_name in enumerate(class_names):
        predictions_df[f'prob_{class_name}'] = test_probs[:, i]

    predictions_df.to_csv(os.path.join(save_dir, 'test_predictions.csv'), index=False)

    print(f"\nEvaluation completed in {time.time() - start_time:.2f} seconds")
    print(f"Detailed results saved to {save_dir}")
    print("="*80)

    return accuracy, precision, recall, f1
