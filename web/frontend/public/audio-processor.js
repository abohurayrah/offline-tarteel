class AudioStreamProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    // Pre-allocate output buffer to avoid GC on audio thread
    this._bufferSize = 4800; // send every 300ms at 16kHz
    this._buffer = new Float32Array(this._bufferSize);
    this._writePos = 0;
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || !input[0]) return true;

    const channelData = input[0];
    const inputSampleRate = sampleRate;
    const outputSampleRate = 16000;
    const ratio = inputSampleRate / outputSampleRate;

    // Compute number of output samples from this input frame
    const outputSamples = Math.floor(channelData.length / ratio);

    for (let i = 0; i < outputSamples; i++) {
      // Linear interpolation between adjacent input samples
      const pos = i * ratio;
      const lo = Math.floor(pos);
      const hi = Math.min(lo + 1, channelData.length - 1);
      const frac = pos - lo;
      const sample = channelData[lo] * (1 - frac) + channelData[hi] * frac;

      this._buffer[this._writePos] = sample;
      this._writePos++;

      if (this._writePos >= this._bufferSize) {
        // RMS normalization to target -20 dBFS (RMS ≈ 0.1)
        let sumSq = 0;
        for (let j = 0; j < this._bufferSize; j++) {
          sumSq += this._buffer[j] * this._buffer[j];
        }
        const rms = Math.sqrt(sumSq / this._bufferSize);
        const targetRms = 0.1;

        // Only normalize if there's meaningful signal (avoid amplifying silence)
        if (rms > 0.001) {
          const gain = Math.min(targetRms / rms, 10.0);
          if (Math.abs(gain - 1.0) > 0.05) {
            for (let j = 0; j < this._bufferSize; j++) {
              this._buffer[j] *= gain;
              // Clamp to [-1, 1] to prevent clipping
              if (this._buffer[j] > 1.0) this._buffer[j] = 1.0;
              else if (this._buffer[j] < -1.0) this._buffer[j] = -1.0;
            }
          }
        }

        // Copy to transferable buffer and send
        const chunk = new Float32Array(this._buffer);
        this.port.postMessage(chunk.buffer, [chunk.buffer]);
        this._writePos = 0;
      }
    }

    return true;
  }
}

registerProcessor("audio-stream-processor", AudioStreamProcessor);
