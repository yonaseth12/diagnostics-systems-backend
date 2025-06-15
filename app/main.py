from flask import Flask, request, jsonify
from model.utils import translate_to_english, translate_to_local_language
from model.city_model.debre_markos_model import predict_disease_dm
from model.city_model.mojo_model import predict_disease_mj
from model.city_model.addis_zemen_model import predict_disease_az
from model.constants import disease_classes
from model.suggestion_layer import disease_advice, THRESHOLDS, get_top_disease_and_suggestion

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
  data = request.get_json()
  user_location = data.get('user_location')
  user_language = data.get('user_language')
  note = data.get('note')
  age = data.get('age', 30)
  gender = data.get('gender', 'male')
  body_temp = data.get('body_temp', 37)
  systolic = data.get('systolic', 120)
  diastolic = data.get('diastolic', 90)

  note = translate_to_english(note, user_language)

  prediction = None
  # Select model based on user_location
  if user_location == 'Debre Markos':
    prediction = predict_disease_dm(note, age, gender, body_temp, systolic, diastolic)
  elif user_location == 'Mojo':
    prediction = predict_disease_mj(note, age, gender, body_temp, systolic, diastolic)
  elif user_location == 'Addis Zemen':
    prediction = predict_disease_az(note, age, gender, body_temp, systolic, diastolic)
  else:
    prediction = predict_disease_dm(note, age, gender, body_temp, systolic, diastolic)

  disease_name, suggestion = get_top_disease_and_suggestion(prediction)

  suggestion_translated = translate_to_local_language(suggestion, user_language)
  disease_name_translated = translate_to_local_language(disease_name, user_language)

  return jsonify({
    "disease": disease_name_translated,
    "suggestion": suggestion_translated,
    "predictions": prediction
  })

if __name__ == '__main__':
  app.run(debug=True) 