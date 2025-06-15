# Thresholds for classification
THRESHOLDS = {
    "high": 0.75,
    "medium": 0.45
}



def get_top_disease_and_suggestion(predictions):
  # predictions: list of (disease, probability) tuples, sorted by probability desc
  top_disease, top_prob = predictions[0]
  if top_prob >= THRESHOLDS['high']:
    severity = 'high'
  elif top_prob >= THRESHOLDS['medium']:
    severity = 'medium'
  else:
    severity = 'low'
  suggestion = disease_advice.get(top_disease, {}).get(severity, "No suggestion available.")
  return top_disease, suggestion



# Suggestion dictionary keyed by disease name and severity level
disease_advice = {
    "Malaria": {
        "high": "High risk of malaria detected. Seek immediate medical attention and begin antimalarial treatment.",
        "medium": "Moderate risk of malaria. Monitor for fever, chills, and body aches. Consider testing.",
        "low": "Symptoms suggest mild risk of malaria. Stay hydrated and consult a doctor if fever persists."
    },
    "Diabetes": {
        "high": "Indicators strongly suggest diabetes. Get a blood sugar test and consult an endocrinologist.",
        "medium": "Signs may point to pre-diabetes. Consider lifestyle changes and monitor glucose levels.",
        "low": "Low probability of diabetes. Maintain a balanced diet and regular activity."
    },
    "Cholera": {
        "high": "Cholera likely. Urgent rehydration therapy and antibiotics are recommended.",
        "medium": "Cholera risk present. Watch for diarrhea and dehydration. Ensure clean water intake.",
        "low": "Low cholera risk. Maintain hygiene and stay hydrated."
    },
    "Meningitis": {
        "high": "High risk of meningitis. Seek emergency medical care immediately.",
        "medium": "Some symptoms match meningitis. Watch for neck stiffness and fever. Visit a clinic if worsens.",
        "low": "Low meningitis likelihood. Remain alert for severe neurological symptoms."
    },
    "Polio": {
        "high": "Possible polio detected. Contact health services immediately.",
        "medium": "Signs suggest polio exposure. Ensure vaccination status is up to date.",
        "low": "Low risk of polio. Keep vaccinations current and stay cautious."
    },
    "Hypertension": {
        "high": "Elevated risk of hypertension. Begin monitoring blood pressure regularly.",
        "medium": "Mild signs of high blood pressure. Adjust salt intake and manage stress.",
        "low": "Blood pressure appears normal. Continue healthy lifestyle practices."
    },
    "Hepatitis": {
        "high": "Strong signs of hepatitis. Get liver function tests and consult a specialist.",
        "medium": "Possible hepatitis signs. Monitor fatigue and jaundice symptoms.",
        "low": "Hepatitis unlikely. Maintain liver health with safe food and hygiene."
    },
    "Typhoid": {
        "high": "Typhoid likely. Seek medical attention for diagnostic confirmation and antibiotics.",
        "medium": "Moderate typhoid probability. Stay hydrated and monitor digestive symptoms.",
        "low": "Low typhoid risk. Ensure food and water safety."
    },
    "Typhus": {
        "high": "Symptoms point to typhus. Seek immediate antibiotic treatment.",
        "medium": "Some typhus indicators present. Avoid insect bites and maintain hygiene.",
        "low": "Typhus risk low. Continue personal hygiene and clean surroundings."
    },
    "TB": {
        "high": "High tuberculosis risk. Consult for sputum test and chest X-ray urgently.",
        "medium": "Possible TB exposure. Monitor for chronic cough and fatigue.",
        "low": "TB unlikely. Maintain good ventilation and immunity."
    },
    "Pneumonia": {
        "high": "Pneumonia strongly suspected. Begin antibiotic therapy after evaluation.",
        "medium": "Mild pneumonia signs. Stay warm and monitor respiratory symptoms.",
        "low": "Low pneumonia risk. Watch for persistent cough or chest pain."
    },
    "Epilepsy": {
        "high": "Likely epilepsy. Medical supervision and anti-epileptic treatment needed.",
        "medium": "Potential seizure activity. Neurological evaluation recommended.",
        "low": "Low epilepsy likelihood. Monitor if further symptoms develop."
    },
    "Diarheal Diseases": {
        "high": "Diarrheal disease detected. Oral rehydration and medication needed urgently.",
        "medium": "Moderate risk. Hydrate well and eat bland foods.",
        "low": "Mild digestive distress possible. Observe and avoid contaminated food."
    },
    "Goiter": {
        "high": "Goiter likely. Check thyroid hormone levels and consider iodine supplementation.",
        "medium": "Thyroid issues may be present. Ultrasound and labs advised.",
        "low": "Low goiter risk. Maintain sufficient iodine in diet."
    },
    "Measles": {
        "high": "Measles detected. Isolate and seek immediate medical care.",
        "medium": "Potential measles. Monitor rash and fever closely.",
        "low": "Low measles risk. Ensure vaccination is up to date."
    },
    "Kwashiorkor/Marasmus": {
        "high": "Severe malnutrition detected. Begin nutritional rehabilitation urgently.",
        "medium": "Moderate malnutrition risk. Improve protein and calorie intake.",
        "low": "Nutrition level appears adequate. Continue balanced diet."
    },
    "Scurvy": {
        "high": "High likelihood of scurvy. Increase vitamin C intake immediately.",
        "medium": "Moderate signs of vitamin C deficiency. Add citrus and vegetables.",
        "low": "Scurvy unlikely. Maintain varied diet."
    },
    "Severe Malnutrition": {
        "high": "Severe malnutrition suspected. Consult for inpatient nutritional support.",
        "medium": "Moderate malnutrition signs. Improve nutrient density in meals.",
        "low": "Low malnutrition risk. Ensure caloric needs are met."
    },
    "Other": {
        "high": "Condition unclear. Seek further medical evaluation.",
        "medium": "Symptoms do not match known diseases. Clinical consultation advised.",
        "low": "Diagnosis inconclusive. Monitor symptoms and visit a physician."
    }
}
