import pyaudio
import wave
import numpy as np

# Configuración de parámetros
fs = 44100  # Frecuencia de muestreo (Hz)
channels = 2  # Número de canales de audio
chunk = 1024  # Tamaño del buffer de audio
threshold = 500  # Umbral para la detección de sonido
silence_duration = 2  # Duración del silencio (segundos) para detener la grabación
output_file = "output.wav"  # Nombre del archivo de salida

def is_silent(data):
    """Determina si el audio está por debajo del umbral de silencio."""
    return np.max(np.abs(data)) < threshold

def capture_audio():
    """Captura audio desde el micrófono hasta que se detecte silencio prolongado."""
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=channels,
                    rate=fs,
                    input=True,
                    frames_per_buffer=chunk)

    print("Esperando para iniciar grabación...")

    frames = []
    recording = False
    silent_chunks = 0

    while True:
        data = stream.read(chunk)
        audio_data = np.frombuffer(data, dtype=np.int16)

        if is_silent(audio_data):
            if recording:
                silent_chunks += 1
                if silent_chunks > (silence_duration * fs / chunk):
                    break
            else:
                continue
        else:
            if not recording:
                print("Comenzando grabación...")
                recording = True
            silent_chunks = 0

        frames.append(data)

    print("Grabación finalizada.")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(output_file, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(fs)
    wf.writeframes(b''.join(frames))
    wf.close()

    print(f"Archivo guardado como {output_file}")

if __name__ == "__main__":
    capture_audio()
