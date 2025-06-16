# Cricket-Team-Prediction-Using-ML
Cricket Team Prediction using Machine Learning Developed a predictive model using K-Means clustering and association rule mining to select optimal ODI cricket teams based on player stats and match conditions. Achieved 88% accuracy and deployed a Flask web app for real-time predictions. Tools: Python, Pandas, Scikit-learn, Matplotlib, mlxtend, Flask

To run this app:
 docker build -t cricket-recommender .  
 docker run -p 5000:5000 cricket-recommender
