from flask import Flask, request, jsonify
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Create Flask app
app = Flask(__name__)

# Load the model and vectorizer using raw string or double backslashes
with open(r"C:\Users\PC\model_and_vectorizer.pkl", 'rb') as f:
    loaded_objects = pickle.load(f)

svm = loaded_objects['svm']
vectorizer = loaded_objects['vectorizer']

# Load the job data
df = pd.read_csv(r"C:\Users\PC\dice_com-job_us_sample.csv")  

# Define your recommend_jobs function
def recommend_jobs(user_input, vectorizer, svm, df):
    pred = vectorizer.transform([user_input.lower()])
    output = svm.predict(pred)
    response = "You may look into " + output[0] + ' jobs\n'

    cos = []
    label_data = df[df['label'] == output[0]]

    for index, row in label_data.iterrows():
        skills = [row['skills']]
        skill_vec = vectorizer.transform(skills)
        cos_lib = cosine_similarity(skill_vec, pred)
        cos.append(cos_lib[0][0])

    label_data['cosine_similarity'] = cos
    top_5 = label_data.sort_values("cosine_similarity", ascending=False).head(5)
    
    for index, row in top_5.iterrows():
        response += f"\nCompany: {row['company']}\n"
        response += f"Job Title: {row['jobtitle']}\n"
        response += f"Location: {row['joblocation_address']}\n"
        response += f"Employment Type: {row['employmenttype_jobstatus']}\n"
        response += f"Job Description: {row['jobdescription']}\n"

    return response

# Define a route for the root URL
@app.route('/')
def home():
    return "Welcome to the Job Recommendation API. Use the /recommend endpoint to get job recommendations."

# Define a route for the favicon
@app.route('/favicon.ico')
def favicon():
    return '', 204  # Return a "No Content" response

# Define a route for your recommend_jobs function
@app.route('/recommend', methods=['POST'])
def recommend():
    user_input = request.json['user_input']
    result = recommend_jobs(user_input, vectorizer, svm, df)
    return jsonify({"result": result})

if __name__ == '__main__':
    app.run(debug=True)
