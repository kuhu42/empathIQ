from pymongo import MongoClient
from datetime import datetime
from angle_calc import angle_calc  # Import your angle_calc function

# Establish a connection to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["posture_analysis"]
collection = db["best_posture_scores"]

def store_best_score(rula_score, reba_score):
    """
    Store the RULA and REBA scores in MongoDB.
    """
    if str(rula_score).isdigit() and str(reba_score).isdigit():
        data = {
            "rula_score": int(rula_score),
            "reba_score": int(reba_score),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        # Insert the document into the collection
        collection.insert_one(data)
        print("Scores stored successfully:", data)
    else:
        print("Invalid scores. Scores were not stored.")

def process_output(output_string):
    """
    Processes the output string from your posture detection model and stores the scores.
    """
    scores = output_string.split()
    rula_score = scores[0]
    reba_score = scores[1]

    store_best_score(rula_score, reba_score)

# Example Usage: Replace this with your actual output string from the posture detection model.
# In a real-time scenario, you would call this function whenever you receive new output.
output_string = "4 3 3 3 3 3 3 3 3 3 3 3 3 3 3 4 3 4 3 4 NULL 4 NULL 3 NULL 3 NULL 3" #Example, your output will vary.

process_output(output_string)

# In your real-time loop, call process_output with the new output string each time.
# Example (conceptual):
# while True:
#     new_output = get_posture_detection_output() # Replace with your actual function to get the detection output.
#     process_output(new_output)
#     time.sleep(1)  # Or whatever delay is appropriate.