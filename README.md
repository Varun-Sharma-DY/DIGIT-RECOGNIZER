# Neural Digit Recognizer

A real-time handwritten digit recognition app using a neural network trained on MNIST.

---

## SETUP & RUN (Do this in order)

### Step 1 — Open the project folder in VS Code terminal

```
cd DIGIT-RECOGNIZER
```

### Step 2 — Create a virtual environment (recommended)

```
python -m venv venv
```

Activate it:
- **Windows:**  `venv\Scripts\activate`
- **Mac/Linux:** `source venv/bin/activate`

### Step 3 — Install dependencies

```
pip install -r requirements.txt
```

> This installs TensorFlow, Flask, and NumPy. May take 2–3 minutes.

### Step 4 — Train the model (one-time, ~2 minutes)

```
python train_model.py
```

This downloads MNIST and trains the neural network. You'll see accuracy ~98%.
It saves a file called `digit_model.h5`.

### Step 5 — Run the app

```
python app.py
```

### Step 6 — Open in browser

Go to: **http://127.0.0.1:5000**

Draw a digit on the canvas and click **ANALYZE**.

---

## Project Structure

```
digit-recognizer/
├── train_model.py      ← Train & save the neural network
├── app.py              ← Flask backend (serves UI + prediction API)
├── digit_model.h5      ← Saved model (created after training)
├── requirements.txt    ← Python dependencies
├── templates/
│   └── index.html      ← Flashy frontend UI
└── README.md
```

## Model Architecture

- Input: 784 neurons (28×28 flattened image)
- Hidden layer 1: 512 neurons (ReLU) + Dropout 20%
- Hidden layer 2: 256 neurons (ReLU) + Dropout 20%
- Hidden layer 3: 128 neurons (ReLU)
- Output: 10 neurons (Softmax → digit probabilities)

Achieves ~98% accuracy on MNIST test set.
