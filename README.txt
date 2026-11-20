HEALTHCARE ANALYTICS PROJECT - HOW TO RUN

Files:
- Healthcare_Analytics_End_to_End.ipynb : complete Pandas, EDA and four-model notebook
- app.py : Streamlit interactive dashboard and patient prediction interface
- Healthcare_Cleaned.csv : supplied dataset
- requirements.txt : Python dependencies

1) Install Python 3.10 or 3.11 (recommended for broad package compatibility).
2) Open a terminal in this folder:
   python -m venv .venv
   .venv\\Scripts\\activate          (Windows PowerShell)
   python -m pip install -r requirements.txt

3) Open the notebook:
   jupyter notebook Healthcare_Analytics_End_to_End.ipynb
   Keep Healthcare_Cleaned.csv in the same folder as the notebook.

4) Launch the Streamlit app:
   streamlit run app.py

The Streamlit app trains the four models on first launch if saved artifacts are not present, then selects the highest ROC-AUC model. This first run may take several minutes.

IMPORTANT PROJECT SCOPE:
The provided CSV contains High_Risk_Flag (High Risk / Standard), not readmitted_30_days. The app predicts High_Risk_Flag only. Do not report this as 30-day readmission prediction unless a true readmission outcome column is added and validated. The dataset's definition of High_Risk_Flag should be confirmed with your mentor. This is an academic demonstration, not a clinical tool.
