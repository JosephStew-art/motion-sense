import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from dtaidistance import dtw
import numpy as np
from dtaidistance import clustering
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage
from tslearn.preprocessing import TimeSeriesScalerMinMax
from scipy.cluster.hierarchy import dendrogram

def get_ds_infos():
    """
    Read the file includes data subject information.
    
    Data Columns:
    0: code [1-24]
    1: weight [kg]
    2: height [cm]
    3: age [years]
    4: gender [0:Female, 1:Male]
    
    Returns:
        A pandas DataFrame that contains inforamtion about data subjects' attributes 
    """ 

    dss = pd.read_csv("data/data_subjects_info.csv")
    print("[INFO] -- Data subjects' information is imported.")
    
    return dss

def set_data_types(data_types=["userAcceleration"]):
    """
    Select the sensors and the mode to shape the final dataset.
    
    Args:
        data_types: A list of sensor data type from this list: [attitude, gravity, rotationRate, userAcceleration] 

    Returns:
        It returns a list of columns to use for creating time-series from files.
    """
    dt_list = []
    for t in data_types:
        if t != "attitude":
            dt_list.append([t+".x",t+".y",t+".z"])
        else:
            dt_list.append([t+".roll", t+".pitch", t+".yaw"])

    return dt_list


def creat_time_series(dt_list, act_labels, trial_codes, mode="raw", labeled=True):
    """
    Args:
        dt_list: A list of columns that shows the type of data we want.
        act_labels: list of activites
        trial_codes: list of trials
        mode: It can be "raw" which means you want raw data
        for every dimention of each data type,
        [attitude(roll, pitch, yaw); gravity(x, y, z); rotationRate(x, y, z); userAcceleration(x,y,z)].
        or it can be "mag" which means you only want the magnitude for each data type: (x^2+y^2+z^2)^(1/2)
        labeled: True, if we want a labeld dataset. False, if we only want sensor values.

    Returns:
        It returns a time-series of sensor data.
    
    """
    num_data_cols = len(dt_list) if mode == "mag" else len(dt_list*3)

    if labeled:
        dataset = np.zeros((0,num_data_cols+7)) # "7" --> [act, code, weight, height, age, gender, trial] 
    else:
        dataset = np.zeros((0,num_data_cols))
        
    ds_list = get_ds_infos()
    
    mean_vector = ds_list[['weight', 'height', 'age', 'gender']].mean()

    # Step 2: Compute the Euclidean distance of each subject's metadata from the mean vector
    def euclidean_distance(row):
        return np.linalg.norm(row - mean_vector)

    ds_list['distance_to_mean'] = ds_list[['weight', 'height', 'age', 'gender']].apply(euclidean_distance, axis=1)

    # Step 3: Find the row with the smallest distance
    most_average_subject = ds_list.loc[ds_list['distance_to_mean'].idxmin()]

    # Optional: Drop the temporary distance column
    ds_list.drop(columns='distance_to_mean', inplace=True)

    # Result:
    print("Most average subject:")
    print(most_average_subject)

    print("[INFO] -- Creating Time-Series")
    for sub_id in ds_list["code"]:
        for act_id, act in enumerate(act_labels):
            for trial in trial_codes[act_id]:
                fname = 'data/A_DeviceMotion_data/'+act+'_'+str(trial)+'/sub_'+str(int(sub_id))+'.csv'
                raw_data = pd.read_csv(fname)
                raw_data = raw_data.drop(['Unnamed: 0'], axis=1)
                vals = np.zeros((len(raw_data), num_data_cols))
                for x_id, axes in enumerate(dt_list):
                    if mode == "mag":
                        vals[:,x_id] = (raw_data[axes]**2).sum(axis=1)**0.5        
                    else:
                        vals[:,x_id*3:(x_id+1)*3] = raw_data[axes].values
                    vals = vals[:,:num_data_cols]
                if labeled:
                    lbls = np.array([[act_id,
                            sub_id-1,
                            ds_list["weight"][sub_id-1],
                            ds_list["height"][sub_id-1],
                            ds_list["age"][sub_id-1],
                            ds_list["gender"][sub_id-1],
                            trial          
                           ]]*len(raw_data))
                    vals = np.concatenate((vals, lbls), axis=1)
                dataset = np.append(dataset,vals, axis=0)
    cols = []
    for axes in dt_list:
        if mode == "raw":
            cols += axes
        else:
            cols += [str(axes[0][:-2])]
            
    if labeled:
        cols += ["act", "id", "weight", "height", "age", "gender", "trial"]
    
    dataset = pd.DataFrame(data=dataset, columns=cols)
    return dataset
#________________________________


ACT_LABELS = ["dws","ups", "wlk", "jog", "std", "sit"]
TRIAL_CODES = {
    ACT_LABELS[0]:[1,2,11],
    ACT_LABELS[1]:[3,4,12],
    ACT_LABELS[2]:[7,8,15],
    ACT_LABELS[3]:[9,16],
    ACT_LABELS[4]:[6,14],
    ACT_LABELS[5]:[5,13]
}

## Here we set parameter to build labeld time-series from dataset of "(A)DeviceMotion_data"
## attitude(roll, pitch, yaw); gravity(x, y, z); rotationRate(x, y, z); userAcceleration(x,y,z)
sdt = ["attitude", "userAcceleration", "rotationRate", "gravity"]
print("[INFO] -- Selected sensor data types: "+str(sdt))    
act_labels = ACT_LABELS [0:6]
print("[INFO] -- Selected activites: "+str(act_labels))    
trial_codes = [TRIAL_CODES[act] for act in act_labels]
dt_list = set_data_types(sdt)
dataset = creat_time_series(dt_list, act_labels, trial_codes, mode="raw", labeled=False)
print("[INFO] -- Shape of time-Series dataset:"+str(dataset.shape))    


# row_counts = dataset.groupby('act').size()
# plt.figure(figsize=(8, 5))
# plt.bar(row_counts.index, row_counts.values, tick_label=ACT_LABELS, color='steelblue')
# plt.xlabel('Activity')
# plt.ylabel('Number of Samples')
# plt.title('Number of Samples per Activity Class')
# plt.grid(axis='y', linestyle='--', alpha=0.7)
# plt.tight_layout()
# plt.show()

# series_length = list()
# for act in range(0,6):
#     for sub in range(0,24):
#         series_length.append(dataset[(dataset['id']==sub) & (dataset['act']==act)].shape[0])
        
# plt.figure(2)
# plt.title('Length of Raw Time Series per Subject per Activity')
# plt.xlabel('Length of Time Series')
# plt.ylabel('Frequency')
# plt.hist(series_length, rwidth=0.5, align='left', color='steelblue')
# plt.grid(axis='y', linestyle='--', alpha=0.6)
# plt.tight_layout()
# plt.show()

# Compute correlation matrix
corr = dataset.corr()

# Create heatmap
plt.figure(figsize=(12, 10))

# Customize font and color map for better contrast
sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="RdBu_r",  # More professional diverging colormap
    square=True,
    linewidths=0.5,  # Add separation lines between cells
    linecolor='white',
    cbar_kws={"shrink": 0.8, "aspect": 30, "label": "Correlation Coefficient"},
    annot_kws={"size": 10}
)

# Use LaTeX-style font sizes and font families
plt.title("Correlation Heatmap", fontsize=18, fontweight='bold', pad=11)
plt.xticks(fontsize=12, rotation=45, ha='right')
plt.yticks(fontsize=12, rotation=0)

plt.tight_layout()
plt.show()


# # Define feature names and nice plot titles
# features = ['userAcceleration', 'gravity', 'attitude.ro', 'rotationRate']
# titles = ['User Acceleration', 'Gravity', 'Attitude', 'Rotation Rate']

# # Prepare the figure
# fig, axes = plt.subplots(2, 2, figsize=(16, 12))
# axes = axes.flatten()

# for i, feature in enumerate(features):
#     series_list = []
#     labels_list = []

#     for act in range(6):
#         df = dataset[(dataset['id'] == 2) & (dataset['act'] == act)]

#         if len(df) >= 512:
#             # Extract 3D data (x, y, z) and form a multivariate time series
#             s = df[feature].values[:512]
            
#             if np.all(np.isfinite(s)):
#                 series_list.append(s)
#                 labels_list.append(act_labels[act])
#             else:
#                 print(f"Skipping {feature} act {act}: NaNs or Infs")
#         else:
#             print(f"Skipping {feature} act {act}: too short")

#     if len(series_list) < 2:
#         print(f"Not enough valid data for {feature}, skipping dendrogram.")
#         continue

#     # Compute DTW distance matrix
#     ds = dtw.distance_matrix(series_list)

#     # Clean up the matrix
#     ds[~np.isfinite(ds)] = 0
#     np.fill_diagonal(ds, 0)
#     ds_condensed = squareform(ds)

#     # Perform clustering
#     Z = linkage(ds_condensed, method='average')

#     # Plot dendrogram
#     dendrogram(
#         Z,
#         labels=labels_list,
#         leaf_rotation=0,
#         leaf_font_size=13,
#         color_threshold=0.4 * max(Z[:, 2]),
#         orientation='left',
#         ax=axes[i]
#     )
#     axes[i].set_title(f"{titles[i]}", fontsize=14)

# # Super title and shared labels
# fig.suptitle("Hierarchical Clustering of Activities Using DTW Across All Features", fontsize=20, y=0.98)
# fig.text(0.5, 0.01, 'Distance', ha='center', fontsize=16)
# fig.text(0.02, 0.5, 'Activity', va='center', rotation='vertical', fontsize=16)

# plt.tight_layout(rect=[0.05, 0.05, 1, 0.98])  # Leave space for title and axis labels
# plt.show()

# from tslearn.preprocessing import TimeSeriesScalerMinMax
# from tslearn.clustering import TimeSeriesKMeans
# from tslearn.metrics import dtw
# from sklearn.manifold import TSNE
# import numpy as np

# series_list = []
# label_list = []

# fixed_length = 400
# features = ['userAcceleration.z']

#             # 'attitude.roll', 'attitude.pitch', 'attitude.yaw',
#             # 'userAcceleration.x', 'userAcceleration.y', 'userAcceleration.z',
#             # 'rotationRate.x', 'rotationRate.y', 'rotationRate.z',
#             # 'gravity.x', 'gravity.y', 'gravity.z'
            
# for act in range(6):  # or use act_labels if needed
#     for sub in dataset['id'].unique():
#         df = dataset[(dataset['id'] == sub) & (dataset['act'] == act)]
#         if len(df) >= fixed_length:
#             segment = df[features].iloc[:fixed_length].to_numpy()
#             if np.all(np.isfinite(segment)):
#                 series_list.append(segment)
#                 label_list.append(act)  # Or use act_labels[act] for string labels

# X = np.array(series_list)  # Shape: (n_samples, time_steps, n_features)

# # ➋ Normalize each feature to [0, 1] range
# scaler = TimeSeriesScalerMinMax()
# X_scaled = scaler.fit_transform(X)

# # ➌ Cluster using DTW-based KMeans
# model = TimeSeriesKMeans(n_clusters=6, metric="dtw", verbose=True, random_state=42)
# y_pred = model.fit_predict(X_scaled)

# # ➍ Flatten for t-SNE: reshape each time series into a 1D vector
# n_samples, time_steps, n_features = X_scaled.shape
# X_flat = X_scaled.reshape((n_samples, time_steps * n_features))

# # ➎ Apply t-SNE
# tsne = TSNE(n_components=2, perplexity=30, random_state=42)
# X_2D = tsne.fit_transform(X_flat)

# from matplotlib.lines import Line2D
# from scipy.spatial import ConvexHull

# plt.figure(figsize=(10,7))
# scatter = plt.scatter(X_2D[:, 0], X_2D[:, 1], c=label_list, cmap='tab10', s=50, alpha=0.6)
# plt.title("t-SNE Plot with Convex Hulls per Activity")
# plt.xlabel("t-SNE 1")
# plt.ylabel("t-SNE 2")
# plt.grid(True)

# colors = plt.cm.tab10.colors  # 10 distinct colors

# for i, act_label in enumerate(ACT_LABELS):
#     points = X_2D[np.array(label_list) == i]
#     if len(points) >= 3:  # Convex hull needs at least 3 points
#         hull = ConvexHull(points)
#         hull_points = points[hull.vertices]
#         # Close the polygon by repeating first point at the end
#         hull_points = np.concatenate([hull_points, hull_points[:1]], axis=0)
#         plt.plot(hull_points[:, 0], hull_points[:, 1], color=colors[i], lw=2, label=f'{act_label} hull')

# # Scatter legend for points (activities)
# legend_elements = [Line2D([0], [0], marker='o', color='w', label=ACT_LABELS[i],
#                           markerfacecolor=colors[i], markersize=10) for i in range(len(ACT_LABELS))]
# plt.legend(handles=legend_elements, title="Activity", bbox_to_anchor=(1.05, 1), loc='upper left')

# plt.tight_layout()
# plt.show()