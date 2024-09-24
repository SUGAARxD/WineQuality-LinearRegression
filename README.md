# WineQuality-LinearRegression

## Description

This project implements two variants of a simple linear regression model using a Multi-Layer Perceptron (MLP). The goal is to predict wine quality scores based on their caracteristics.

## Technologies

The first implementation is built using `Numpy` and the second uses `PyTorch`.

## Architecture Details

- The model consists of **3 layers of neurons**. The first layer is the input layer that has 11 neurons for the 11 features of a wine, the hidden layer has 20 neurons, but it can be modified, and the output layer has one neuron that represents the predicted score. 

- **Activation Functions:** The model uses `Leaky ReLU` in the hidded layer and linear function(no activation) on the output.

- **Optimizer and Loss Function:** For the numpy implementation I use `Gradient Descent` and for the pytorch implementation I use the `Adam` optimizer. Both use `Mean Squared Error (MSE)` as loss function.

- **Regularization:** I use `weight decay` as a form of regularization.

- **Evaluation Metrics:** The model's performance is evaluated using `loss` on the test set.

## Dataset

The dataset used for training is [Wine Quality](https://archive.ics.uci.edu/dataset/186/wine+quality). It contains 4,898 instances of wine, each with 11 physicochemical features (e.g., alcohol, acidity) and a quality score ranging from 0 to 10.

The project has a script to split the data into train, val and test.

## How to Use

### Installation

1. **Clone the repository**
2. **Create and Activate a Virtual Environment Using Conda:**
   ```bash
   conda create --name env_name python=3.10
   ```
   ```bash
   conda activate env_name
   ```
   Replace `env_name` with your desired environment name.
   
4. **Install the Packages and Dependencies:**

   **Using Conda:**
   ```bash
   conda install tensorboard numpy=1.26.4 pandas tqdm pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
   ```

   **Or Using Pip:**

   ```bash
   pip install tensorboard numpy=1.26.4 pandas tqdm torch torchvision
   ```
### Running the code

You can use a code editor like Visual Studio Code or an IDE like PyCharm. Use your environment with all packages installed.

Alternatively, you can run the script from the console:

1. **Enter the Project Folder:**
   For the numpy version:
   ```bash
   cd path_to_repo\WineQuality-LinearRegression\numpy_regression
   ```
   For the pytorch version:
   ```bash
   cd path_to_repo\WineQuality-LinearRegression\torch_regression
   ```
   **Make sure to replace `path_to_repo` with the actual path to your cloned repository.**
3. **Run the Code:**
   ```bash
   python linear_regression.py
   ```
   
   **Or**

   ```bash
   python3 linear_regression.py
   ```
   If the first command does not work.
