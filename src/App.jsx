import { useState, useRef, useEffect } from 'react';

function App() {
  const [image, setImage] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [predictedText, setPredictedText] = useState('');
  const [originalText, setOriginalText] = useState('');
  const [metrics, setMetrics] = useState(null);
  const fileInputRef = useRef(null);

  // Levenshtein Distance
  const levenshteinDistance = (s1, s2) => {
    if (s1.length < s2.length) return levenshteinDistance(s2, s1);
    if (s2.length === 0) return s1.length;

    let previousRow = Array(s2.length + 1).fill(0).map((_, i) => i);
    for (let i = 0; i < s1.length; i++) {
      let currentRow = [i + 1];
      for (let j = 0; j < s2.length; j++) {
        const insertions = previousRow[j + 1] + 1;
        const deletions = currentRow[j] + 1;
        const substitutions = previousRow[j] + (s1[i] !== s2[j] ? 1 : 0);
        currentRow.push(Math.min(insertions, deletions, substitutions));
      }
      previousRow = currentRow;
    }
    return previousRow[previousRow.length - 1];
  };

  // Normalize Kannada Text
  const normalizeKannadaText = (text) => {
    return text.replace(/\s+/g, ' ').trim();
  };

  // Tokenize Kannada Words
  const tokenizeKannadaWords = (text) => {
    const words = text.match(/[ಅ-ೌ]+|[a-zA-Z]+|\d+/g) || [];
    return words.filter(word => word.trim());
  };

  // Calculate CER
  const calculateCER = (original, predicted) => {
    const originalNorm = normalizeKannadaText(original);
    const predictedNorm = normalizeKannadaText(predicted);
    const editDistance = levenshteinDistance(originalNorm, predictedNorm);
    const totalChars = originalNorm.length;
    const cer = totalChars === 0 ? (predictedNorm.length === 0 ? 0 : Infinity) : (editDistance / totalChars) * 100;
    return { percentage: cer, editDistance, totalChars };
  };

  // Calculate WER
  const calculateWER = (original, predicted) => {
    const originalNorm = normalizeKannadaText(original);
    const predictedNorm = normalizeKannadaText(predicted);
    const originalWords = tokenizeKannadaWords(originalNorm);
    const predictedWords = tokenizeKannadaWords(predictedNorm);
    const editDistance = levenshteinDistance(originalWords, predictedWords);
    const totalWords = originalWords.length;
    const wer = totalWords === 0 ? (predictedWords.length === 0 ? 0 : Infinity) : (editDistance / totalWords) * 100;
    return { percentage: wer, editDistance, totalWords };
  };

  // Calculate Metrics
  const calculateMetrics = (original, predicted) => {
    const cer = calculateCER(original, predicted);
    const wer = calculateWER(original, predicted);
    return {
      CER: {
        percentage: Number(cer.percentage.toFixed(2)),
        editDistance: cer.editDistance,
        totalCharacters: cer.totalChars,
      },
      WER: {
        percentage: Number(wer.percentage.toFixed(2)),
        editDistance: wer.editDistance,
        totalWords: wer.totalWords,
      },
    };
  };

  // Handle Calculate CER and WER
  const handleCalculateMetrics = () => {
    if (!originalText || !predictedText) {
      setError('Both original and predicted text are required to calculate CER and WER');
      setMetrics(null);
      return;
    }
    setError(null);
    const metricsResult = calculateMetrics(originalText, predictedText);
    setMetrics(metricsResult);
  };

  useEffect(() => {
    if (result) {
      console.log("Result updated:", result);
      if (result.predicted_text_url) {
        fetch(`http://localhost:5000${result.predicted_text_url}`)
          .then((res) => res.text())
          .then((text) => setPredictedText(text))
          .catch((err) => console.error('Failed to fetch predicted text:', err));
      }
    }
  }, [result]);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setImage(URL.createObjectURL(file));
      setResult(null);
      setError(null);
      setPredictedText('');
      setOriginalText('');
      setMetrics(null);
      uploadImage(file);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file) {
      setImage(URL.createObjectURL(file));
      setResult(null);
      setError(null);
      setPredictedText('');
      setOriginalText('');
      setMetrics(null);
      uploadImage(file);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const uploadImage = async (file) => {
    setIsLoading(true);
    const formData = new FormData();
    formData.append('file', file);
    try {
      const response = await fetch('http://localhost:5000/upload', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      console.log('Server Response:', data);
      if (data.error) {
        setError(data.error);
      } else {
        setResult(data);
      }
    } catch (err) {
      setError('Failed to process image');
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-100 p-6">
      <h1 className="text-3xl font-bold text-center mb-6">Kannada OCR App</h1>

      {/* File Upload */}
      <div
        className="border-2 border-dashed border-gray-400 rounded-lg p-6 mb-6 text-center"
        onDrop={handleDrop}
        onDragOver={handleDragOver}
      >
        <input
          type="file"
          accept="image/*"
          onChange={handleFileChange}
          ref={fileInputRef}
          className="hidden"
        />
        <p className="text-gray-600 mb-4">Drag and drop an image here or click to select</p>
        <button
          onClick={() => fileInputRef.current.click()}
          className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600"
        >
          Select Image
        </button>
      </div>

      {/* Error Message */}
      {error && (
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-6">
          {error}
        </div>
      )}

      {/* Loading Indicator */}
      {isLoading && (
        <div className="text-center mb-6">
          <p className="text-lg">Processing image...</p>
        </div>
      )}

      {/* Results */}
      {result && (
        <div className="space-y-6">
          {/* Predicted and Original Text - First */}
          <div className="flex flex-col md:flex-row gap-6">
            <div className="flex-1">
              <h2 className="text-xl font-semibold mb-2">Predicted Text</h2>
              <pre className="bg-white p-4 rounded shadow whitespace-pre-wrap min-h-32">
                {predictedText || 'No text available'}
              </pre>
            </div>
            <div className="flex-1">
              <h2 className="text-xl font-semibold mb-2">Original Text</h2>
              <textarea
                value={originalText}
                onChange={(e) => setOriginalText(e.target.value)}
                className="w-full bg-white p-4 rounded shadow border border-gray-300 focus:outline-none focus:ring-2 focus:ring-blue-500 min-h-32"
                rows="5"
                placeholder="Enter the original text here..."
                aria-label="Original Text Input"
              />
            </div>
          </div>

          {/* Calculate CER/WER Button - Second */}
          <div className="text-center">
            <button
              onClick={handleCalculateMetrics}
              className="bg-blue-500 text-white px-6 py-3 rounded hover:bg-blue-600 text-lg font-semibold"
            >
              Calculate CER and WER
            </button>
          </div>

          {/* CER/WER Results */}
          {metrics && (
            <div>
              <h2 className="text-xl font-semibold mb-2">CER and WER Metrics</h2>
              <div className="bg-white p-4 rounded shadow">
                <p className="mb-2">
                  <strong>CER:</strong> {metrics.CER.percentage}% (Edit distance: {metrics.CER.editDistance}/{metrics.CER.totalCharacters})
                </p>
                <p>
                  <strong>WER:</strong> {metrics.WER.percentage}% (Edit distance: {metrics.WER.editDistance}/{metrics.WER.totalWords})
                </p>
              </div>
            </div>
          )}

          {/* Original Image - Third */}
          <div>
            <h2 className="text-xl font-semibold mb-2">Original Image</h2>
            <img
              src={image}
              alt="Uploaded"
              className="border border-gray-300 max-w-full h-auto rounded shadow"
            />
          </div>

          {/* Predictions Table */}
          <div>
            <h2 className="text-xl font-semibold mb-2">Predicted Characters</h2>
            <div className="overflow-x-auto">
              <table className="w-full border-collapse border border-gray-300">
                <thead>
                  <tr className="bg-gray-200">
                    <th className="border border-gray-300 p-2">Line</th>
                    <th className="border border-gray-300 p-2">Word</th>
                    <th className="border border-gray-300 p-2">Char</th>
                    <th className="border border-gray-300 p-2">Type</th>
                    <th className="border border-gray-300 p-2">Label</th>
                    <th className="border border-gray-300 p-2">Combined</th>
                    <th className="border border-gray-300 p-2">Image</th>
                  </tr>
                </thead>
                <tbody>
                  {result.predictions.map((pred, index) => (
                    <tr key={index} className="hover:bg-gray-50">
                      <td className="border border-gray-300 p-2">{pred.line}</td>
                      <td className="border border-gray-300 p-2">{pred.word}</td>
                      <td className="border border-gray-300 p-2">{pred.char}</td>
                      <td className="border border-gray-300 p-2">{pred.type}</td>
                      <td className="border border-gray-300 p-2">{pred.label}</td>
                      <td className="border border-gray-300 p-2">{pred.combined_char}</td>
                      <td className="border border-gray-300 p-2">
                        <img
                          src={`http://localhost:5000${pred.char_url}?t=${Date.now()}`}
                          alt="Character"
                          className="w-16 h-16 object-contain"
                          onError={(e) => console.error(`Image load failed: ${e.target.src}`)}
                        />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Metrics */}
          <div>
            <h2 className="text-xl font-semibold mb-2">Metrics</h2>
            <div className="bg-white p-4 rounded shadow">
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <p><strong>Total Lines:</strong> {result.metrics.total_lines}</p>
                <p><strong>Total Words:</strong> {result.metrics.total_words}</p>
                <p><strong>Total Characters:</strong> {result.metrics.total_chars}</p>
                <p><strong>Total Ottaksharas:</strong> {result.metrics.total_ottaksharas}</p>
                <p><strong>Average Character Area:</strong> {result.metrics.avg_char_area.toFixed(2)}</p>
                <p><strong>Total Predictions:</strong> {result.metrics.total_predictions}</p>
                <p><strong>Average Confidence:</strong> {result.metrics.avg_confidence.toFixed(2)}</p>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;