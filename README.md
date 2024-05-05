# UFC-Predictor

## What is this application?

This application is mostly for learning Supervised Learning Techniques along with trying to predict which fighter will win in a given UFC MMA match. Note that this data is not accurate (meaning that using this fighter stats are from Feburary 2024, even if a particular match happend in May 2020, we have future data which means these results are not sutable for sports betting), meaning we have a time series problem. But this is a problem with the data and not the models themselves.

## Install Dependencies

Make sure you have the right downloads to run the models by running the code chunk in the **'InstallDependencies.ipynb'** file. Afterwards, you should be set!

## Where do I start?

A great place to start when it comes to looking at the various supervised learning algorithms and the problem itself is by looking at the **'MainDisplay.ipynd'** file. In this file it will train up four common supervised learning models and compare their accuracy along with the number of True Positives, False Positves, True Negatives, and False Negatives.

## How do I check out an individual model?

To check out one model you can visit one of the 4 files, of which each holds a specific model.

- **'LogisticRegression.ipynb'** This File Holds the Logistic Regression Model.
- **'NeuralNetworks.ipynb'** This File Holds the Neural Network Model.
- **'DecisionTree.ipynb'** This File Holds the Decision Tree Model.
- **'RandomForest.ipynb'** This File Holds the Random Forest Model.

## Model Evaluation

### Model Metrics

<table align="center">
  <tr>
    <td>
      <div>Logistic Regression Metrics</div>
      <img src="Images/Metrics/LogisticRegressionMetrics.png" alt="Logistic Regression Metrics" width="80%">
    </td>
    <td>
      <div>Neural Network Metrics</div>
      <img src="Images/Metrics/NeuralNetworkMetrics.png" alt="Neural Network Metrics" width="80%">
    </td>
  </tr>
  <tr>
    <td>
      <div>Decision Tree Metrics</div>
      <img src="Images/Metrics/DecisionTreeMetrics.png" alt="Decision Tree Metrics" width="80%">
    </td>
    <td>
      <div>Random Forest Metrics</div>
      <img src="Images/Metrics/RandomForestMetrics.png" alt="Random Forest Metrics" width="80%">
    </td>
  </tr>
</table>

### Model ROC Curves

<table align="center">
  <tr>
    <td>
      <div>Logistic Regression ROC Curve</div>
      <img src="Images/ROC_Curves/LogisticRegressionCurve.png" alt="Logistic Regression ROC Curve" width="80%">
    </td>
    <td>
      <div>Neural Network ROC Curve</div>
      <img src="Images/ROC_Curves/NeuralNetworkCurve.png" alt="Neural Network ROC Curve" width="80%">
    </td>
  </tr>
  <tr>
    <td>
      <div>Decision Tree ROC Curve</div>
      <img src="Images/ROC_Curves/DecisionTreeCurve.png" alt="Decision Tree ROC Curve" width="80%">
    </td>
    <td>
      <div>Random Forest ROC Curve</div>
      <img src="Images/ROC_Curves/RandomForestCurve.png" alt="Random Forest ROC Curve" width="80%">
    </td>
  </tr>
</table>

### Model Confusion Matrices

<table align="center">
  <tr>
    <td>
      <div>Logistic Regression Confusion Matrix</div>
      <img src="Images/ConfusionMatrices/LogisticRegressionMatrix.png" alt="Logistic Regression Confusion Matrix" width="80%">
    </td>
    <td>
      <div>Neural Network Confusion Matrix</div>
      <img src="Images/ConfusionMatrices/NeuralNetworkMatrix.png" alt="Neural Network Confusion Matrix" width="80%">
    </td>
  </tr>
  <tr>
    <td>
      <div>Decision Tree Confusion Matrix</div>
      <img src="Images/ConfusionMatrices/DecisionTreeMatrix.png" alt="Decision Tree Confusion Matrix" width="80%">
    </td>
    <td>
      <div>Random Forest Confusion Matrix</div>
      <img src="Images/ConfusionMatrices/RandomForestMatrix.png" alt="Random Forest Confusion Matrix" width="80%">
    </td>
  </tr>
</table>

### Visualize Model Comparisons through ROC Curve

![ROC Curve of all models](/Images/ROC_Curve.png "ROC Curve")
