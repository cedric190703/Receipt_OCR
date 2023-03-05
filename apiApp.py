import cv2
from flask import Flask, request, jsonify
from python_app import *

# Create an instance of the application
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def process_image():
    # Vérifie si le fichier image est présent dans la requête
    if 'image' not in request.files:
        return jsonify({'error': 'Aucun fichier image n\'a été envoyé.'}), 400

    # Read the image file from the request and convert it to a NumPy array using cv2.imdecode()
    image_file = request.files['image']
    image_data = image_file.read()
    nparr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Call the main function
    result = main(image)

    # Renvoie les résultats sous forme de fichier JSON
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug = True)