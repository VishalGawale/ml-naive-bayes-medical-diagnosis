# Diagnosing Inflammation Diseases with Naive Bayes & Machine Learning  
Hey there! Welcome to my project, where I use a Naive Bayes classifier to sniff out inflammation-related diseases like nephritis or urinary tract infections. I took a dataset packed with symptoms think temperature spikes, nausea, and nagging pain and built a model to predict what’s going on. It’s a bite sized machine learning adventure with a medical twist, and I’m excited to walk you through it!

## Project Overview  
This is my stab at diagnosing inflammation diseases using machine learning. I leaned on a Naive Bayes classifier to analyze symptoms and figure out if someone’s dealing with something like nephritis or a urinary tract infection. The idea? Turn data into answers fast and smart.

## Tasks & Steps  
Here’s how I made it happen:  
- **Data Prep**: Loaded a dataset with symptoms like `temperature`, `nausea`, and `lumbar_pain`, plus a target label `disease` (mixing inflammation and nephritis). Cleaned it up and got it ready to roll.  
- **Model Time**: Trained a Naive Bayes classifier simple, speedy, and surprisingly sharp for this job.  
- **Visuals & Checks**: Used plots (thanks, Matplotlib and Seaborn!) to see how symptoms tie to diagnoses, and ran some metrics to prove it works.  

## Key Insights  
- Naive Bayes nailed it—quick predictions with solid accuracy for symptom-based diagnosis.  
- Symptoms like `temperature` and `lumbar_pain` were big red flags super telling for spotting trouble.  
- Small project, big potential—think medical apps or triage tools. Recruiters, this could scale!

## Requirements  
Here’s what you’ll need to run this:  
- Python 3.x  
- Libraries: `pandas`, `scikit-learn`, `matplotlib`,
  
## Install dependencies:
```bash
pip install pandas scikit-learn matplotlib seaborn

git clone https://github.com/VishalGawale/ml-naive-bayes-medical-diagnosis.git
cd ml-naive-bayes-medical-diagnosis
pip install -r requirements.txt
