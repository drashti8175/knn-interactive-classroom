# 🚀 KNN Interactive Classroom - Deployment Guide

## 📋 Project Overview

**KNN Interactive Classroom** is an interactive, browser-based educational tool for exploring K-Nearest Neighbors (KNN) classification in 2D, 3D, and 4D dimensions. Built with Python and Streamlit, it provides real-time visualizations and predictions for machine learning students.

---

## 🎯 Key Features

| Feature | Purpose |
|---------|---------|
| **2D Visualization** | Classic scatter plots + decision boundary heatmaps (inverse-distance vs Gaussian weighting) |
| **3D Visualization** | Interactive Plotly 3D scatter with rotation and zoom |
| **4D Visualization** | 3D axes + marker size encodes the 4th feature dimension |
| **Live Prediction** | Set values via sliders → instant KNN class prediction |
| **3 Voting Methods** | Uniform, Inverse-distance, Gaussian kernel weighting |
| **Auto-Select K** | Leave-one-out cross-validation finds optimal K |
| **CSV Upload** | Support for custom datasets with auto-cleaning |
| **Neighbor Table** | Shows selected K neighbors, distances, and vote weights |

---

## 📦 Tech Stack

| Component | Version |
|-----------|---------|
| **Python** | 3.10+ |
| **Streamlit** | 1.36.0+ |
| **Pandas** | 2.0.0+ |
| **NumPy** | 1.24.0+ |
| **Matplotlib** | 3.7.0+ |
| **Scikit-learn** | 1.3.0+ |
| **Plotly** | 5.18.0+ |

---

## 🏗️ Project Structure

```
knn-interactive-classroom/
├── app.py                              # Main Streamlit application
├── requirements.txt                    # Python dependencies
├── README.md                           # User documentation
├── LICENSE                             # MIT License
├── students.csv                        # Sample dataset (root)
├── data/
│   └── students.csv                    # Student dataset (4 features, 28 rows)
├── utils/
│   ├── __init__.py
│   └── knn_utils.py                    # Core KNN algorithms & utilities
└── __init__.py
```

**Dataset Details:**
- **Rows:** 28 student records
- **Features:** Study Hours, Attendance, Assignments, Sleep Hours
- **Target:** Pass/Fail classification (binary)

---

## 🔧 Dependencies

### Core Requirements
- **streamlit** – Web UI framework
- **pandas** – Data manipulation & CSV handling
- **numpy** – Numerical computations
- **matplotlib** – 2D plotting
- **scikit-learn** – Distance metrics & preprocessing
- **plotly** – Interactive 3D/4D visualizations

### Optional (for deployment)
- **gunicorn** – WSGI server (for cloud deployments)
- **python-dotenv** – Environment variable management

---

## 📤 Deployment Options

### **Option 1: Streamlit Cloud (Recommended for beginners)**

**Pros:** Free, simple, automatic updates
**Steps:**
1. Push code to GitHub
2. Visit [streamlit.io/cloud](https://streamlit.io/cloud)
3. Sign in with GitHub
4. Select repository and `app.py`
5. Click "Deploy"
6. App is live in ~2-3 minutes

**URL Format:** `https://<username>-knn-interactive-classroom-<random>.streamlit.app`

**Requirements:**
- Public GitHub repository
- `requirements.txt` in root directory

---

### **Option 2: Heroku**

**Pros:** Good for production apps, custom domain support
**Steps:**
1. Create `Procfile`:
   ```
   web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
   ```

2. Create `setup.sh`:
   ```bash
   mkdir -p ~/.streamlit/
   echo "[server]
   headless = true
   port = $PORT
   enableCORS = false
   " > ~/.streamlit/config.toml
   ```

3. Deploy:
   ```bash
   heroku login
   heroku create your-app-name
   git push heroku main
   ```

**Cost:** Free tier available (~$5/month for paid)

---

### **Option 3: AWS (Elastic Beanstalk)**

**Pros:** Scalable, professional infrastructure
**Steps:**
1. Install AWS CLI & EB CLI
2. Create `requirements.txt` (already have it)
3. Deploy:
   ```bash
   eb init -p python-3.10 knn-app
   eb create knn-env
   eb deploy
   ```

**Cost:** ~$5-20/month depending on usage

---

### **Option 4: PythonAnywhere**

**Pros:** Simple Python hosting
**Steps:**
1. Sign up at [pythonanywhere.com](https://www.pythonanywhere.com)
2. Upload files via web interface
3. Create bash console, install requirements
4. Start Streamlit app from Web tab

**Cost:** Free tier available (~$5/month for paid)

---

### **Option 5: Docker + Self-hosted (Advanced)**

**Create `Dockerfile`:**
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
```

**Build & Run:**
```bash
docker build -t knn-app .
docker run -p 8501:8501 knn-app
```

---

## ✅ Pre-Deployment Checklist

- [ ] Python 3.10+ installed locally
- [ ] All dependencies in `requirements.txt`
- [ ] `app.py` runs without errors: `streamlit run app.py`
- [ ] `data/students.csv` exists and is readable
- [ ] No hardcoded API keys or secrets in code
- [ ] Code tested with sample dataset
- [ ] README.md updated with deployment URL
- [ ] LICENSE file included
- [ ] `.gitignore` configured properly
- [ ] Git repository initialized and pushed

---

## 🧪 Local Testing (Before Deployment)

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run locally
streamlit run app.py

# 4. Test in browser at http://localhost:8501
# - Load default data
# - Try 2D, 3D, 4D visualizations
# - Test predictions
# - Upload custom CSV
# - Auto-select K
```

---

## 🌍 Recommended Deployment Path

**For fastest deployment (< 5 minutes):**
1. ✅ Push to GitHub
2. ✅ Connect to Streamlit Cloud
3. ✅ Share public link with users

**Next steps if more traffic:**
- Upgrade to Streamlit Cloud Pro ($10/month)
- Or migrate to Heroku/AWS for scaling

---

## 📊 Expected Performance

| Metric | Value |
|--------|-------|
| Startup time | ~3-5 seconds |
| 2D plot rendering | < 500ms |
| 3D/4D interactive | 60 FPS |
| Cross-validation (K search) | ~2-5 seconds (28-row dataset) |
| CSV upload/processing | < 2 seconds |

---

## 🔒 Security Considerations

- ✅ No database required (CSV-based)
- ✅ No authentication needed (public educational tool)
- ✅ User-uploaded CSVs processed in memory (not stored)
- ✅ No sensitive data exposure
- ✅ HTTPS automatic on Streamlit Cloud

---

## 📞 Troubleshooting Deployment

| Issue | Solution |
|-------|----------|
| "Module not found" | Ensure `requirements.txt` has all imports |
| App takes too long to load | Increase Streamlit Cloud compute tier |
| 3D plots lag | Reduce data size or use GPU tier |
| CSV upload fails | Check file format (UTF-8, no special chars) |
| Custom domain needed | Use Streamlit Cloud Pro or Heroku |

---

## 📝 Environment Variables (Optional)

If needed, create `.streamlit/config.toml`:
```toml
[server]
headless = true
port = 8501
enableCORS = false

[theme]
primaryColor = "#3b82f6"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"
```

---

## 🎓 Educational Deployment Tips

- **Classroom Use:** Share Streamlit Cloud link with students
- **Assignment:** Have students upload their own datasets
- **Learning Path:** Start with 2D → 3D → 4D → Custom datasets
- **Feedback:** Add GitHub Issues link for student feedback

---

**Last Updated:** 2026-08-16  
**Status:** Ready for Deployment ✅
