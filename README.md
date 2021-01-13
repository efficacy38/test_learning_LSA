# Training a TensorFlow.js model for Speech Commands Using Browser FFT
<img src="./markups/info-markup.svg"></img><br>
## 安裝jupyter-notebook
### 使用 pip 安裝~~
```
sudo apt update
sudo apt install python3-pip
pip3 install --upgrade pip
pip3 install jupyter
pip install -U jupyter
```

==**記得登出再登入，不然會找不到jupyter~**==

### 進到 workdir 開啟 jupyter note book
預設 ~/test_learn
創建資料夾
```
cd ~
mkdir test_learn
```
下載這個專案的 code
安裝 git
```
sudo apt install git
```
clone repo
```
git clone https://github.com/efficacy38/test_learning_LSA.git
cd test_learning_LSA/
```
開啟 jupyter-notebook
```
jupyter notebook
```
點開 training_custom_audio_model_in_python.ipynb

### training_custom_audio_model_in_python.ipynb 注意重點
1. 在training_custom_audio_model_in_python.ipynb
    ```python=
    def resample_wavs(dir_path, target_sample_rate=44100):


      """Resample the .wav files in an input directory to given sampling rate.

      The resampled waveforms are written to .wav files in the same directory with
      file names that ends in "_44100hz.wav".

      44100 Hz is the sample rate required by the preprocessing model. It is also
      the most widely supported sample rate among web browsers and mobile devices.
      For example, see:
      https://developer.mozilla.org/en-US/docs/Web/API/AudioContextOptions/sampleRate
      https://developer.android.com/ndk/guides/audio/sampling-audio

      Args:
        dir_path: Path to a directory that contains .wav files.
        target_sapmle_rate: Target sampling rate in Hz.
      """
      wav_paths = glob.glob(os.path.join(dir_path, "*.wav"))
      resampled_suffix = "_%shz.wav" % target_sample_rate
      for i, wav_path in tqdm.tqdm(enumerate(wav_paths)):
        if wav_path.endswith(resampled_suffix):
          continue
        sample_rate, xs = wavfile.read(wav_path)
        xs = xs.astype(np.float32)
        xs = librosa.resample(xs, sample_rate, TARGET_SAMPLE_RATE).astype(np.int16)
        resampled_path = os.path.splitext(wav_path)[0] + resampled_suffix
        wavfile.write(resampled_path, target_sample_rate, xs)


    for word in WORDS:
      word_dir = os.path.join(DATA_ROOT, word)
      if os.path.isdir(word_dir) or "zh" in word_dir :
          print(word_dir)
          resample_wavs(word_dir, target_sample_rate=TARGET_SAMPLE_RATE)
    ```
    第 33 行裡面我的設定是在 `/tmp/speech_commands_v0.02` 底下任何有檔名包含 zh 都會被 resampling 和變成之後訓練的 data set，所以請把你的音檔按照你想讓他辨識出的名子放在`/tmp/speech_commands_v0.02`並加個zh後墜，他就會在辨識的時候跑出你當初資料夾設定的名子~
2. 當你一直按下一步案到最後，你就可以看到 train 好的 tfjs model 放在 /tmp/tfjs-model 把它整個上傳到 github 就完成了第二步了
3. 設定 tfjs 語音辨識
    在`/dev_web/index.js`
    ```jsx=
    async function app() {
    // recognizer = speechCommands.create('BROWSER_FFT', null, 'https://raw.githubusercontent.com/efficacy38/test_LSA_audio_predict/main/model.json', 'https://raw.githubusercontent.com/efficacy38/test_LSA_audio_predict/main/metadata.json');  en+zh
        recognizer = speechCommands.create('BROWSER_FFT', null, 'https://raw.githubusercontent.com/efficacy38/test_LSA_audio_predict-zh-/main/model.json', 'https://raw.githubusercontent.com/efficacy38/test_LSA_audio_predict-zh-/main/metadata.json');
     await recognizer.ensureModelLoaded();
    }
    ```
    在 app function 中 model.json，和 metadata.json 的部分可以透過上傳model.json, metadata.json,和 group1-shard2of2.bin... 到 github，並且用 github 右上角那個raw按鍵，顯示出只有文檔的網頁，複製連結並照著格式去改完index.js，應該就可以打開 index.html 快樂的辨識囉
