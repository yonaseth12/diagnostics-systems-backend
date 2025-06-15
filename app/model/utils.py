# cleaning, projection, tokenization utilities will go here 

from deep_translator import GoogleTranslator

def translate_to_english(note, user_language):
  if user_language == 'en':
    return note
  if user_language in ['am', 'so', 'om', 'ti']:
    return GoogleTranslator(source=user_language, target='en').translate(note)
  return note

def translate_to_local_language(note, user_language):
  if user_language == 'en':
    return note
  if user_language in ['am', 'so', 'om', 'ti']:
    return GoogleTranslator(source='en', target=user_language).translate(note)
  return note

# Test cases for translation
# notes = {
#   'am': "ዶክተር ማህበሩ በጤና ችግር ላይ ነው።",
#   'so': "Dhakhtarku wuxuu leeyahay bukaanku xanuun ayuu dareemayaa.",
#   'om': "Doqtoorichi dhukkubsataa rakkoo fayyaa qaba jedhe.",
#   'ti': "ዶክተሩ ብሓንቲ ጤና ተጋጊዱ ኣሎ።",
#   'en': "The doctor says the patient is having a health problem."
# }
# languages = ['am', 'so', 'om', 'ti', 'en']

# print("\n--- Local to English ---")
# for lang in languages:
#   if lang != 'en':
#     local_note = notes[lang]
#     translated = translate_to_english(local_note, lang)
#     print(f"{lang.upper()} to EN: {local_note}  -->  {translated}")

# print("\n--- English to Local ---")
# for lang in languages:
#   if lang != 'en':
#     en_note = notes['en']
#     translated = translate_to_local(en_note, lang)
#     print(f"EN to {lang.upper()}: {en_note}  -->  {translated}")
