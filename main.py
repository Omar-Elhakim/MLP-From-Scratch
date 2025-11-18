from sklearn.metrics._classification import confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
import mlp
import pandas as pd
import plotly.express as px
import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


if __name__ == "__main__":
    title = "NN Task 2"
    st.set_page_config(page_title=title)
    st.title(title)

    # Reading the data
    data = pd.read_csv("penguins.csv")

    # print(data["Species"].value_counts())
    # There's 8 values missing , we will impute them by the column mean
    # print(data.notna().sum().sum())

    data.fillna(data.mean(numeric_only=True), inplace=True)

    le = LabelEncoder()
    data["OriginLocation"] = le.fit_transform(data["OriginLocation"])
    data["Species"] = le.fit_transform(data["Species"])

    numeric_cols = data.columns.drop(["OriginLocation", "Species"])
    scaler = StandardScaler()
    data[numeric_cols] = scaler.fit_transform(data[numeric_cols])

    # First 30 samples of every species is taken as training data and the rest is for testing

    trainingData = pd.concat(
        (
            data[data["Species"] == 0][:30],
            data[data["Species"] == 1][:30],
            data[data["Species"] == 2][:30],
        )
    )
    testData = data.drop(trainingData.index)

    # Shuffle the data and reset its index

    trainingData = trainingData.sample(frac=1).reset_index(drop=True)
    testData = testData.sample(frac=1).reset_index(drop=True)

    xTrain, yTrain = (
        trainingData[trainingData.columns.drop("Species")],
        trainingData["Species"],
    )
    xTest, yTest = testData[testData.columns.drop("Species")], testData["Species"]

    hiddenLayers = []
    hlInput = st.text_input("Hidden Layers sizes ex: 10,2,3", value="10")

    hpParameters = st.columns(2)
    with hpParameters[0]:
        lr = float(st.text_input("Learning Rate", value=1e-3))
        st.write(lr)
    with hpParameters[1]:
        epochs = st.number_input("Number of Epochs", value=50)

    act = st.selectbox("**Choose Activation Function**", ["sigmoid", "tanh"])
    useBias = st.checkbox("Bias")
    for s in hlInput.split(","):
        if s == "":
            continue
        hiddenLayers.append(int(s.strip()))

    mlp = mlp.MLP(5, hiddenLayers, 3)
    mlp.lr = lr  # learning rate
    if useBias:
        mlp.UseBias()
    if act == "tanh":
        mlp.UseTanh()
    mlp.Init()

    # Start Trainig
    acc = 0
    accurecies = []

    for epoch in range(epochs):
        acc = mlp.fit(xTrain, yTrain)
        accurecies.append(acc / len(xTrain))

    fig = px.line(accurecies)
    st.plotly_chart(fig)

    # Start Testing
    acc = 0
    yPredict = mlp.predict(xTest)
    for i in range(len(yPredict)):
        acc += 1 if (yPredict[i] == yTest[i]) else 0

    cm = confusion_matrix(yTest, yPredict)
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
    )
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    st.pyplot(fig)

    st.write(f"Train Accuracy: {accurecies[-1] * 100}")
    st.write(f"Test Accuracy: {acc / len(yTest) * 100}")
