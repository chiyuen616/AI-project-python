from sanic import Sanic
from sanic.response import json



# class_names = ['Iris setosa', 'Iris versicolor', 'Iris virginica']
app = Sanic("AnomAlert")

@app.route("/")
def test(request):
    return json({"hello": "world"})

# model = tf.saved_model.load('./')
# @app.post("/")
# def callModel(request):
#     content = request.json

#     predict_dataset = tf.convert_to_tensor(content)

#     predictions = model(predict_dataset, training=False)
#     probs = tf.nn.softmax(predictions)

#     class_indexes = tf.argmax(probs, axis = 1).numpy()
#     results = []

#     for i, class_idx in enumerate(class_indexes):
#         name = class_names[class_idx]
#         p = np.max(probs[i].numpy())
#         results.append({
#             "name": name,
#             "probability": float(p)
#         })

#     return json({ "data": results })



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, single_process=True)