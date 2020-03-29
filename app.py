import json, re, random, requests, redis
import numpy as np
from flask import Flask, request, jsonify
from textgenrnn import textgenrnn

# load config vars, init app and cache
app = Flask(__name__, static_url_path='')
app.config.from_object('config')
cache = redis.from_url(app.config['REDIS_URL'])

# load model
model = textgenrnn(
    weights_path='model_weights.hdf5',
    vocab_path='model_vocab.json',
    config_path='model_config.json',
)
speakers = ['sarge', 'simmons', 'tucker', 'caboose', 'donut']
punct = '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\\n\\t\'‘’“”’–—'


def preprocess(message, speaker):
    text = speaker + ':' + message.lower() + ('' if message and message[-1] in '?!.' else '.')
    text = re.sub('([{}])'.format(punct), r' \1 ', text)
    return ' '.join(text.split())


def generate(conversation):
    prefix = ' '.join(conversation) + ' grif : '
    response = model.generate(prefix=prefix, return_as_list=True)[0]
    response = response[len(prefix):] + ' '
    conversation.append('grif : ' + response[:-1])
    del conversation[:-1]

    response = re.sub(' ([{}]) '.format(punct), r'\1 ', response)
    response = re.sub("' (t|s|d|m|re|ve|ll)([\s{}])".format(punct), r"'\1\2", response)
    response = response.strip() if response.strip() else '...'
    if ':' in response:
        response = response.split(':')[0].rsplit(' ', 1)[0]
    return response


def respond(sender, message):
    # get conversation history from cache
    print('Message from {0}: {1}'.format(sender, message))
    try:
        speaker, conversation = json.loads(cache.get(sender))
    except Exception as e:
        speaker = random.choice(speakers)
        conversation = ['grif : hey , what \' s up ?']
    message = preprocess(message, speaker)
    conversation.append(message)
        
    # send and cache model response
    response = generate(conversation)
    print('Response to {0}: {1}'.format(sender, response))
    cache.set(sender, json.dumps([speaker, conversation]), ex=86400)
    return response


@app.route('/hook', methods=['GET'])
def verify():
    if request.args.get('hub.mode') == 'subscribe' and request.args.get('hub.challenge'):
        if not request.args.get('hub.verify_token') == app.config['VERIFY_TOKEN']:
            return 'Verification token mismatch', 403
        return request.args['hub.challenge']
    return 'Improper verification request'


@app.route('/hook', methods=['POST'])
def webhook():
    for sender, message in messaging_events(request.get_json()):
        response = respond(sender, message)
        send_message(sender, response)
    return 'OK'


@app.route('/', methods=['GET', 'POST'])
def chat():
    # return webpage on GET request
    if request.method == 'GET':
        return app.send_static_file('index.html')

    sender = request.get_json()['id']
    message = request.get_json()['text']
    response = respond(sender, message)
    return jsonify({'text': response})


def messaging_events(data):
    events = data['entry'][0]['messaging']
    for event in events:
        if 'message' in event and 'text' in event['message']:
            yield event['sender']['id'], event['message']['text']


def send_message(recipient, message):
    requests.post(
        'https://graph.facebook.com/v6.0/me/messages',
        params={'access_token': app.config['PAGE_ACCESS_TOKEN']},
        json={
            'messaging_type': 'RESPONSE',
            'recipient': {'id': recipient},
            'message': {'text': message}
        }
    )


if __name__ == '__main__':
    speaker, conversation = random.choice(speakers), []
    while True:
        message = input('Say something: ')
        message = preprocess(message, speaker)
        conversation.append(message)
        response = generate(conversation)
        print(response)
    # app.run(debug=True)
