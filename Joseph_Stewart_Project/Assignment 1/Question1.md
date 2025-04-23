## Based on the information you can gather from the Github page and other related sources please answer the following questions with regards to the (A) DeviceMotion data set:

### a. How many input features are present in this dataset? (1)

Based on the code and documentation, there are 12 input features:
- Attitude (roll, pitch, yaw) - 3 features
- Gravity (x, y, z) - 3 features
- RotationRate (x, y, z) - 3 features
- UserAcceleration (x, y, z) - 3 features

### b. Are these raw sensor values or have there been pre-processing steps? (2)

These are raw sensor values directly from the device's motion sensors. This is evident from:
1. The data structure showing direct sensor readings
2. The presence of separate preprocessing steps in the tutorial notebooks
3. The code showing direct CSV imports of raw sensor data without prior transformation

### c. Are there any input features that you think would not be necessary or are redundant for human activity recognition? (2)

Yes, there are potentially redundant features:
1. Gravity vector might be redundant since it's already partially captured in the attitude measurements (roll, pitch)
2. Some rotation information is represented in both attitude and rotationRate, making them partially redundant for activity recognition purposes

### d. How many different target classes are provided in this dataset for classification? What are they? (2)

From the code, particularly in the tutorial notebooks, there are 6 main activity classes:
- Standing (STR/STD)
- Walking (WAL)
- Jogging (JOG)
- Jumping (JUM)
- Sitting (SIT)
- Falling (FALL)

### e. How many examples are in this dataset if you were to consider each trial an example? (1)

Looking at the trial codes in the repository:
- Downstairs (dws): trials [1,2,11] = 3 trials
- Upstairs (ups): trials [3,4,12] = 3 trials
- Walking (wlk): trials [7,8,15] = 3 trials
- Jogging (jog): trials [9,16] = 2 trials
- Standing (std): trials [6,14] = 2 trials
- Sitting (sit): trials [5,13] = 2 trials

Total number of trials = 15 examples

### f. Suppose you want to increase the number of examples by slicing the trials into fixed windows of time. e.g. instead of one long recording per activity, divide it into multiple smaller recordings. What do you think potential issues with this approach might be? (2)

Potential issues with time-window slicing:
1. Transitions between activities might be split across windows, creating ambiguous or incorrect labels
2. Some activities require longer time contexts to be properly identified (e.g., walking up stairs vs. walking), and small windows might lose this temporal context
3. Overlapping windows could create data leakage between training and testing sets if not carefully separated
