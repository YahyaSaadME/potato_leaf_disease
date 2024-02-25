from flask import Flask, request, jsonify
import tensorflow as tf

app = Flask(__name__)

# Load the model
model = tf.keras.models.load_model("E:/Projects/FDA_PROJECT/backened/plant.h5")
classnames = ['Potato Early blight', 'Potato Late blight', 'Potato healthy']

# Function to preprocess a single image
def preprocess_image(image_path):
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(256, 256))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = tf.expand_dims(image, axis=0)  # Add batch dimension
    image = tf.keras.applications.resnet50.preprocess_input(image)
    return image

# Function to predict a single image
def predict_single_image(image_path, model):
    image = preprocess_image(image_path)
    predictions = model.predict(image)
    predicted_class_idx = tf.argmax(predictions[0]).numpy()
    confidence = predictions[0][predicted_class_idx] * 100
    predicted_class = classnames[predicted_class_idx]
    return predicted_class, confidence

@app.route('/', methods=['GET'])
def home():
    return '''
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FDA PROJECT</title>
</head>

<body>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Lexend:wght@100..900&display=swap');

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'lexend', sans-serif;
        }

        #imagePreview {
            max-width: 300px;
            max-height: 300px;
            margin-top: 20px;
            border-radius: 4px;
        }

        #imageUpload {
            display: none;
            /* Hide the default file input */
        }

        #customButton {
            background-color: #000000;
            color: white;
            padding: 10px 20px;
            border: none;
            cursor: pointer;
            border-radius: 5px;
            width: 60%;
        }

        #customButton:hover {
            background-color: #000000;
        }
    </style>
    <div style="padding: 10px;">
        <h1>Potato Desiese</h1>
        <p style="margin-bottom: 40px;">Upload your potato 🥔 leaf photo and check weather it is affected or not.</p>
        <div style="display: flex; justify-content: center; width: 100%;">
            <div>
                <img id="img" src="./assets/uploadReq.jpg" style="width: 500px;" alt="">
                <div style="display: flex;justify-content: center;">
                    <button id="customButton" onclick="btn()">
                        Choose File
                    </button>
                    <input type="file" accept=".jpg" id="imageUpload" onchange="previewImage(event)">
                    <div id="imagePreview"></div>
                    
                </div>
                    <div id="output-div" style="display:none;padding: 10px;box-shadow: 2px 2px 5px -2px black; border: .5px solid rgb(109, 109, 109);border-radius: 5px;margin-top: 10px;">
                </div>
                    <center>
                    <div id="reset" onclick="reload()" style="cursor: pointer; display:none;background-color: black;border-radius: 5px;margin-top: 10px;color: white;padding: 10px;">reset</div>
                </center>
            </div>
        </div>
    </div>
    </div>

    <script>
            function reload(){
            location.reload()
        }
        async function previewImage(event) {
            var reader = new FileReader();
            reader.onload = function () {
                var img = document.createElement("img");
                img.src = reader.result;
                img.style.maxWidth = "100%";
                img.style.maxHeight = "100%";
                var preview = document.getElementById("imagePreview");
                preview.innerHTML = '';
                preview.appendChild(img);
            }
            reader.readAsDataURL(event.target.files[0]);
            document.getElementById('img').style.display = 'none'
            document.getElementById('customButton').style.display = 'none'
            var formData = new FormData();
            formData.append('image', event.target.files[0]);
            const req = await fetch("/upload",{
                method:"POST",
                body: formData
            })
            const res = await req.json()
            console.log(res)
            if(res){
            document.getElementById("output-div").style.display=null;
            document.getElementById("reset").style.display=null;
            document.getElementById('output-div').innerHTML = `
            <center>
                <h2>Accuracy: ${res.confidence}%</h2>
                <h2>Belongs to  ${res.predicted_class}</h2>
            </center>
            `
            }

        }
        function btn() {
            document.getElementById('imageUpload').click();
        }

    </script>
</body>

</html>


'''

@app.route('/upload', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image found in request'}), 400

    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'No image selected for uploading'}), 400

    # Save the image to a temporary location
    temp_image_path = "temp_image.jpg"
    image_file.save(temp_image_path)

    # Make prediction
    predicted_class, confidence = predict_single_image(temp_image_path, model)

    # Return the result
    result = {
        'predicted_class': predicted_class,
        'confidence': round(confidence,2)
    }
    return jsonify(result), 200

if __name__ == '__main__':
    app.run()
