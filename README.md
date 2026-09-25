# MNIST Digit Classifier — Deployment

A Streamlit app that classifies hand-drawn or uploaded digit images using a CNN trained on MNIST.

## 1. Train the model (run locally, needs internet to download MNIST)

```bash
pip install tensorflow
python train_model.py
```

This creates `model.h5` in the same folder. Training takes a few minutes on CPU.

## 2. Test the app locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL Streamlit prints (usually `http://localhost:8501`) and confirm predictions work.

## 3. Push everything to GitHub

Create a new repo (e.g. `mnist-deployment`) and push these files:

```
app.py
train_model.py
requirements.txt
model.h5
README.md
```

```bash
git init
git add app.py train_model.py requirements.txt model.h5 README.md
git commit -m "MNIST classifier app"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

> `model.h5` for this small CNN is only a few MB, so it's fine to commit directly (no Git LFS needed).

## 4. Deploy on Streamlit Community Cloud (free, fastest)

1. Go to https://share.streamlit.io and sign in with your GitHub account.
2. Click **New app**.
3. Select your repository, branch (`main`), and main file path (`app.py`).
4. Click **Deploy**.
5. Wait 1–3 minutes for the build to finish. You'll get a live link like:
   `https://<your-app-name>.streamlit.app`

## 5. Submit

Submit both:
- **Deployment link**: the `*.streamlit.app` URL from step 4
- **GitHub repo link**: `https://github.com/<your-username>/<your-repo>`

Before submitting, open the live link yourself and make sure a prediction actually returns a digit.
"# mnist-deployment" 
