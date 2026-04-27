import os
import sys
import tempfile

import librosa
import soundfile as sf
from flask import Flask, request, send_file, render_template

sys.path.insert(0, os.path.dirname(__file__))
from src.models import UpgradedUNet
from src.audio_utils import get_spec, spec_to_wav, process_overlap_add, apply_agc

import torch

app = Flask(__name__)

# 앱 시작 시 모델을 한 번만 로드 (Singleton)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UpgradedUNet().to(device)

MODEL_PATH = '/workspace/model/upgraded_gan_model.pth'

if os.path.exists(MODEL_PATH):
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['generator_state_dict'])
    model.eval()
    print(f'모델 로드 완료: {MODEL_PATH}')
else:
    print(f'경고: {MODEL_PATH} 가중치 파일이 없습니다.')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/process_audio', methods=['POST'])
def process_audio_api():
    if 'file' not in request.files:
        return {'error': '파일이 제공되지 않았습니다.'}, 400

    file = request.files['file']
    if file.filename == '':
        return {'error': '선택된 파일이 없습니다.'}, 400

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_in, \
             tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_out:

            file.save(temp_in.name)

            y, _ = librosa.load(temp_in.name, sr=16000)
            denoised_y = process_overlap_add(y, model, device)
            final_y = apply_agc(denoised_y)
            sf.write(temp_out.name, final_y, 16000)

            response = send_file(temp_out.name, as_attachment=True,
                                 download_name='denoised_output.wav')

            @response.call_on_close
            def cleanup():
                if os.path.exists(temp_in.name):
                    os.remove(temp_in.name)
                if os.path.exists(temp_out.name):
                    os.remove(temp_out.name)

            return response

    except Exception as e:
        return {'error': str(e)}, 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
