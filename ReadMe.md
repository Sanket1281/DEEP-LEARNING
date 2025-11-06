# Food Vision with PyTorch: The Impact of Data Augmentation

This project demonstrates the effectiveness of data augmentation in improving the performance and generalization of a Convolutional Neural Network (CNN). Two models are trained on a food image dataset to classify images of pizza, steak, and sushi: one without data augmentation and one with it. The results clearly show that the augmented model achieves better accuracy and is less prone to overfitting.

The core of this project is a custom CNN architecture, "TinyVGG," built using PyTorch.

---

## 🚀 Key Objectives

* To build a custom CNN model (**TinyVGG**) from scratch using PyTorch.
* To train a baseline model (**Model 0**) on a food image dataset without any data augmentation.
* To train a second model (**Model 1**) using the same architecture but with data augmentation (`TrivialAugmentWide`).
* To compare the performance of both models by visualizing their training/testing loss and accuracy curves.

---

## Dataset

The project uses the **Pizza, Steak, and Sushi** subset of the Food101 dataset, which can be found on Kaggle. The data is structured into `train` and `test` directories, with subdirectories for each class:

```
data/
└── pizza_steak_sushi/
    ├── train/
    │   ├── pizza/
    │   ├── steak/
    │   └── sushi/
    └── test/
        ├── pizza/
        ├── steak/
        └── sushi/
```

---

## 🛠️ Models & Techniques

The project centers around comparing two models with identical architectures but different training data transformations.

### 1. Model Architecture: TinyVGG

A simple yet effective CNN architecture is implemented, featuring three convolutional blocks followed by a fully connected classifier.

* **Convolutional Block:** Each block consists of:
    1.  `nn.Conv2d`
    2.  `nn.ReLU`
    3.  `nn.Conv2d`
    4.  `nn.ReLU`
    5.  `nn.MaxPool2d`

* **Classifier:**
    1.  `nn.Flatten`
    2.  `nn.Linear` (to produce logits for the output classes)

### 2. Model 0: Baseline (No Data Augmentation)

This model is trained on images that are only resized and converted to tensors.
* [cite_start]**Transforms:** `transforms.Resize()` and `transforms.ToTensor()`. [cite: 67, 68, 69, 70]

### 3. Model 1: With Data Augmentation

This model is trained on the same data, but with a powerful augmentation technique applied to the training set.
* [cite_start]**Transforms:** `transforms.Resize()`, `transforms.TrivialAugmentWide()`, and `transforms.ToTensor()`. [cite: 346]

---

## 📊 Results & Comparison

The performance of both models was tracked over 30 epochs. The comparison plots clearly show the benefits of data augmentation:

* **Lower Overfitting:** Model 1 (with augmentation) shows a much smaller gap between its training and testing accuracy/loss, indicating better generalization.
* **Higher Test Accuracy:** Model 1 consistently achieves higher accuracy on the test dataset, demonstrating its ability to perform well on unseen data.
* **More Stable Training:** The loss curve for Model 1 is generally more stable than that of the baseline model.


---

## ⚙️ How to Run

### Prerequisites

Make sure you have Python and the following libraries installed:

* `torch`
* `torchvision`
* `pandas`
* `matplotlib`
* `tqdm`

You can install them using pip:
```bash
pip install torch torchvision pandas matplotlib tqdm
```

### Steps

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-name>
    ```

2.  **Download the data:**
    Download the dataset and structure it as described in the [Dataset](#dataset) section.

3.  **Run the Notebook:**
    Open and run the `Data-Augmentation.ipynb` notebook in a Jupyter environment. The notebook contains all the code for data loading, model building, training, and result visualization.