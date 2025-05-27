import seaborn as sns 
import matplotlib.pyplot as plt
import numpy as np

class_names = ['Downstairs', 'Upstairs', 'Sitting', 'Standing', 'Walking', 'Jogging']

cm8_2 = np.array([
    [0.81481481, 0.13580247, 0.        , 0.        , 0.04938272, 0.        ],
    [0.08181818, 0.87272727, 0.00909091, 0.02727273, 0.00909091, 0.        ],
    [0.        , 0.        , 1.        , 0.        , 0.        , 0.        ],
    [0.        , 0.        , 0.        , 1.        , 0.        , 0.        ],
    [0.09756098, 0.        , 0.        , 0.        , 0.90243902, 0.        ],
    [0.0106383 , 0.0106383 , 0.        , 0.        , 0.        , 0.9787234 ]
])

cm2_5 = np.array([
    [0.77066667, 0.11733333, 0.        , 0.        , 0.112     , 0.        ],
    [0.07279693, 0.86590038, 0.        , 0.05938697, 0.00191571, 0.        ],
    [0.        , 0.        , 1.        , 0.        , 0.        , 0.        ],
    [0.        , 0.        , 0.        , 1.        , 0.        , 0.        ],
    [0.07475083, 0.00996678, 0.        , 0.        , 0.90780731, 0.00747508],
    [0.        , 0.        , 0.        , 0.        , 0.00886918, 0.99113082]
])

plt.figure(figsize=(20, 8))

plt.subplot(1, 2, 1)
sns.heatmap(cm8_2, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('True', fontsize=12)
plt.title('Normalised Confusion Matrix: Model 8.2', fontsize=14, fontweight='bold')

plt.subplot(1, 2, 2)
sns.heatmap(cm2_5, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('')
plt.title('Normalised Confusion Matrix: Model 2.5', fontsize=14, fontweight='bold')

# Improve layout
plt.tight_layout()
plt.show()