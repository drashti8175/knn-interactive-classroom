<div align="center">

<h1>🎓 KNN Interactive Classroom</h1>

<p>
  <strong>An interactive, browser-based tool to explore K-Nearest Neighbors classification<br/>
  in 2D, 3D, and 4D — built with Python & Streamlit.</strong>
</p>

<p>
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Streamlit-1.36%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/Plotly-Interactive-3D8FD9?style=for-the-badge&logo=plotly&logoColor=white"/>
  <img src="https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge"/>
</p>

<p>
  <a href="#-features">Features</a> •
  <a href="#-demo">Demo</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-how-it-works">How It Works</a> •
  <a href="#-project-structure">Project Structure</a> •
  <a href="#-upload-your-own-csv">Your Own CSV</a> •
  <a href="#-tech-stack">Tech Stack</a>
</p>

</div>

---

## 🌟 Features

| Feature | Description |
|---|---|
| **2D Visualization** | Classic Matplotlib scatter + decision boundary heatmap comparing inverse-distance vs Gaussian weighting |
| **3D Visualization** | Fully interactive Plotly 3-D scatter — drag to rotate, scroll to zoom, hover for values |
| **4D Visualization** | 3-D axes + **marker size** encodes a 4th feature dimension |
| **Live Prediction** | Set any point via sliders → instantly see which class KNN predicts |
| **3 Voting Methods** | Uniform · Inverse-distance · Gaussian kernel weighting |
| **Auto-Select K** | Leave-one-out cross-validation finds the best K automatically |
| **Upload Any CSV** | Bring your own labelled dataset — missing values are auto-cleaned |
| **Decision Boundary** | Side-by-side boundary maps for inverse-distance vs Gaussian (2D mode) |
| **Neighbor Table** | See exactly which K rows were selected, their distances, and vote weights |

---

## 🎬 Demo

> Run locally in under 60 seconds — see [Quick Start](#-quick-start) below.

```
streamlit run app.py
```

The app loads a built-in **student dataset** (28 rows, 4 features: Study Hours, Attendance, Assignments, Sleep Hours → Pass/Fail).

**Try this workflow:**
1. Open the app → it shows the 2D scatter with the default data
2. Switch to **3D** in the sidebar → rotate the chart by dragging
3. Set Study_Hours = `6`, Attendance = `80`, Assignments = `75` → click **🚀 Predict class**
4. Switch to **4D** to see Sleep_Hours encoded as marker size
5. Enable **Auto-select K** to run cross-validation and pick the best K

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/drashti8175/knn-interactive-classroom.git
cd knn-interactive-classroom
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run app.py
```

The browser opens automatically at **http://localhost:8501**

---

## 🧠 How It Works

KNN classifies a new point by finding its **K closest neighbors** in the training data and letting them vote.

### Algorithm steps

```
1. Scale     →  Min-max normalize all features to [0, 1]
2. Distance  →  Compute scaled Euclidean distance to every training row
3. Neighbors →  Select the K rows with the smallest distances
4. Vote      →  Each neighbor votes for its class (3 weighting modes)
5. Predict   →  Class with the highest total score wins
```

### Distance formula

$$d(p,q) = \sqrt{\sum_{j=1}^{n} \left(\frac{p_j - q_j}{M_j - m_j}\right)^2}$$

Where $m_j$ and $M_j$ are the dataset min and max for feature $j$.

### Voting methods

| Method | Weight formula | Effect |
|---|---|---|
| **Uniform** | `w = 1` | Every neighbor counts equally |
| **Inverse-distance** | `w = 1 / d` | Closer neighbors count more |
| **Gaussian** | `w = exp(−d² / 2σ²)` | Smooth decay; σ controls locality |

### Visualization modes

| Mode | Dimensions shown | Chart type |
|---|---|---|
| **2D** | X axis + Y axis | Matplotlib (static) + decision boundary |
| **3D** | X + Y + Z axes | Plotly (interactive — rotate & zoom) |
| **4D** | X + Y + Z + **marker size** | Plotly (interactive) |

---

## 📁 Project Structure

```
knn-interactive-classroom/
│
├── app.py                  # Streamlit app — UI, plots, sidebar, prediction panel
│
├── utils/
│   ├── __init__.py
│   └── knn_utils.py        # Pure KNN engine — scaling, distance, voting, CV
│
├── data/
│   └── students.csv        # Built-in demo dataset (28 rows, 4 features)
│
├── requirements.txt        # Python dependencies
├── .gitignore
└── README.md
```

### Key functions in `knn_utils.py`

| Function | Purpose |
|---|---|
| `feature_bounds()` | Compute min/max per column for scaling |
| `scale_features()` | Min-max normalize a feature vector |
| `calculate_distance()` | Scaled Euclidean distance between two points |
| `get_nearest_neighbors()` | Find K closest training examples |
| `vote_scores()` | Aggregate weighted votes from K neighbors |
| `predict_result()` | Return the winning class label |
| `cross_validate_k()` | Leave-one-out CV across candidate K values |

---

## 📤 Upload Your Own CSV

Any labelled CSV works. Requirements:

- **Minimum:** 2 numeric feature columns + 1 label column
- **For 3D mode:** at least 3 numeric columns
- **For 4D mode:** at least 4 numeric columns
- **Missing values:** auto-handled — empty columns are dropped, incomplete rows are removed, and a warning tells you exactly what changed
- **Label column:** can be text (`Pass/Fail`, `cat/dog`, `A/B/C`) or numeric

### Example CSV format

```csv
feature_1,feature_2,feature_3,label
5.1,3.5,1.4,setosa
4.9,3.0,1.4,setosa
6.3,3.3,4.7,versicolor
7.1,3.0,5.9,virginica
```

---

## 🛠 Tech Stack

| Library | Version | Used for |
|---|---|---|
| [Streamlit](https://streamlit.io) | ≥ 1.36 | Web UI, widgets, sidebar, caching |
| [Pandas](https://pandas.pydata.org) | ≥ 2.0 | Data loading, cleaning, display |
| [NumPy](https://numpy.org) | ≥ 1.24 | Vector math, distances, Gaussian weights |
| [Matplotlib](https://matplotlib.org) | ≥ 3.7 | 2D scatter + decision boundary |
| [Plotly](https://plotly.com) | ≥ 5.18 | Interactive 3D & 4D charts |
| [scikit-learn](https://scikit-learn.org) | ≥ 1.3 | Available for future extensions |

---

## 📸 App Layout

```
┌─────────────────────────────────────────────────────────┐
│  🎓 KNN Interactive Classroom                           │
├──────────────┬──────────────────────────────────────────┤
│   SIDEBAR    │  MAIN AREA                               │
│              │                                          │
│  📁 Dataset  │  Dataset preview table + class chart     │
│  🔭 Mode     │                                          │
│   2D│3D│4D  │  ─────────────────────────────────────   │
│              │                                          │
│  🗂 Columns  │  Prediction result + vote scores         │
│  📍 New pt   │                                          │
│  K selector  │  K-nearest neighbors table               │
│  Weighting   │                                          │
│  Sigma σ     │  📊 2D / 3D / 4D Visualization           │
│              │                                          │
│ [🚀 Predict] │  Decision boundary comparison (2D)       │
└──────────────┴──────────────────────────────────────────┘
```

---

## 🤝 Contributing

Contributions, issues and feature requests are welcome!

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -m "Add my feature"`
4. Push to the branch: `git push origin feature/my-feature`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 👩‍💻 Author

**Drashti** — [@drashti8175](https://github.com/drashti8175)

> Built with ❤️ using Python + Streamlit
