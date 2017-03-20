import os
from gtts import gTTS
from pygame import mixer


def speak(text, wait=True):
    mixer.quit()
    mixer.init(29000)
    file_path = "Sound/" + text[1] + ".mp3"
    my_file = os.path.isfile(file_path)

    if not my_file:
        speech = gTTS(text=text[0], lang='en')
        speech.save(file_path)
    mixer.music.load(file_path)
    mixer.music.play()
    if wait:
        while mixer.music.get_busy():
            pass
