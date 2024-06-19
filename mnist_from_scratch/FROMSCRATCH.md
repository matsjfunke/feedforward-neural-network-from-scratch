# Steps to coding a Neural Network

1. **Dataset Loading**

2. **Data Preparation:**
   - The dataset (`train.csv`) is converted into a NumPy array and shuffled.
   - The data is split into a development set (`data_dev`) and a training set (`data_train`).
   - For both sets, labels (`Y_dev`, `Y_train`) and features (`X_dev`, `X_train`) are separated. Features are normalized by dividing by 255.

3. **Random Initialization of Parameters**

4. **Forward Propagation:**
    - use input-data, activation functions, weights & biases --> to predict

5. **Calculatr Cost:**
    - evaluate how much to prediction is from the label

6. **Backward Propagation:**
    - use gradients of activation functions to minize the cost function & calc weights & biases

7. **Parameter Update:**
    - update weights & biases according to backward propagation

looping through 4 -> 5 -> 6 -> 7 = **learning**


8. **Use trained model:**
    - use the through training calculated Parameters to predict outside the training example

output of train_test_model.py
(train_test_model.png)
