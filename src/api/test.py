import requests

# The URL of the login and prediction endpoints
login_url = "http://127.0.0.1:3000/login"
predict_url = "http://127.0.0.1:3000/v1/models/admissions/predict"

# Données de connexion
credentials = {
    "username": "user123",
    "password": "password123"
}

# Send a POST request to the login endpoint
login_response = requests.post(
    login_url,
    headers={"Content-Type": "application/json"},
    json=credentials
)

# Check if the login was successful
if login_response.status_code == 200:
    token = login_response.json().get("token")
    print("Token JWT obtenu:", token)

    # Data to be sent to the prediction endpoint
    data = {
        "serial_nb": 362,
        "gre": 334,
        "toefl": 116,
        "university_rating": 4,
        "sop": 4.0,
        "lor" : 3.5,
        "cgpa": 9.54,
        "research": 1
    }

    # Send a POST request to the prediction
    response = requests.post(
        predict_url,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}"
        },
        json=data
    )

    print("Réponse de l'API de prédiction:", response.text)
else:
    print("Erreur lors de la connexion:", login_response.text)
