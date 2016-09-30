# coding=utf-8
import json
from watson_developer_cloud import LanguageTranslationV2


language_translation = LanguageTranslationV2(
    username='YOUR SERVICE USERNAME',
    password='YOUR SERVICE PASSWORD')

# create new custom model
# with open('../resources/language_translation_model.tmx', 'rb') as custom_model:
#     print(json.dumps(language_translation.create_model(
#         base_model_id='en-fr', name='test_glossary', forced_glossary=custom_model), indent=2))

print(json.dumps(language_translation.get_models(), indent=2))

print(json.dumps(language_translation.get_model('en-es-conversational'), indent=2))

# delete custom model
# print(json.dumps(language_translation.delete_model('13860c86-ec3f-4e60-8cbe-3ef0048f92af'), indent=2))

print(json.dumps(language_translation.translate('Hola, cómo estás? €', source='es', target='en'), indent=2,
                 ensure_ascii=False))

print(json.dumps(language_translation.identify('Hello how are you?'), indent=2))


print(json.dumps(language_translation.get_identifiable_languages(), indent=2))
