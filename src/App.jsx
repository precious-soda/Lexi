// Lexi - An OCR for Kannada Handwritten Text
// Copyright (C) 2025  Sahil Kumar Jamwal

// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

// Contact: Sahil Kumar Jamwal
// Email: sahilkumarjamwal464@gmail.com


import { useState, useRef, useEffect } from 'react';
import { Document, Page, pdfjs } from 'react-pdf';
pdfjs.GlobalWorkerOptions.workerSrc = `//cdnjs.cloudflare.com/ajax/libs/pdf.js/${pdfjs.version}/pdf.worker.min.js`;

function App() {
  const [file, setFile] = useState(null);
  const [fileType, setFileType] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [predictedText, setPredictedText] = useState('');
  const [originalText, setOriginalText] = useState('');
  const [metrics, setMetrics] = useState(null);
  const [selectedModel, setSelectedModel] = useState('densenet121_attention');
  const [showMetricsButton, setShowMetricsButton] = useState(false);
  const fileInputRef = useRef(null);

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

  const normalizeKannadaText = (text) => {
    return text.replace(/\s+/g, ' ').trim();
  };

  const tokenizeKannadaWords = (text) => {
    const words = text.match(/[ಅ-ೌ]+|[a-zA-Z]+|\d+/g) || [];
    return words.filter(word => word.trim());
  };

  const calculateCER = (original, predicted) => {
    const originalNorm = normalizeKannadaText(original);
    const predictedNorm = normalizeKannadaText(predicted);
    const editDistance = levenshteinDistance(originalNorm, predictedNorm);
    const totalChars = originalNorm.length;
    const cer = totalChars === 0 ? (predictedNorm.length === 0 ? 0 : Infinity) : (editDistance / totalChars) * 100;
    return { percentage: cer, editDistance, totalChars };
  };

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

  const organizeSegmentationData = (segmentation, predictions) => {
    if (!segmentation?.lines || !Array.isArray(predictions)) {
      console.error('Invalid segmentation or predictions data');
      return {};
    }
    const organized = {};
    Object.entries(segmentation.lines).forEach(([lineId, lineData]) => {
      organized[lineId] = {
        lineData: lineData,
        words: {}
      };
      if (lineData.words) {
        Object.entries(lineData.words).forEach(([wordId, wordData]) => {
          organized[lineId].words[wordId] = {
            wordData: wordData,
            characters: []
          };
        });
      }
    });
    predictions.forEach(pred => {
      if (organized[pred.line] && organized[pred.line].words[pred.word]) {
        organized[pred.line].words[pred.word].characters.push(pred);
      }
    });
    return organized;
  };

  const handleCalculateMetrics = () => {
    if (!originalText.trim() || !predictedText.trim()) {
      setError('Please provide both original and predicted text to calculate metrics.');
      setMetrics(null);
      setShowMetricsButton(false);
      return;
    }
    setError(null);
    const metricsResult = calculateMetrics(originalText, predictedText);
    setMetrics(metricsResult);
    setShowMetricsButton(true); // Show the Metrics button after calculating
  };

  const handleDownloadTranscribedPDF = async () => {
    if (!result?.session_id) {
      setError('No session ID available for download.');
      return;
    }
    try {
      const response = await fetch(`http://localhost:5000/generate_transcribed_pdf/${result.session_id}`, {
        method: 'POST', // Changed to POST to send metrics data
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          metrics: metrics || {}, // Send metrics to backend
        }),
      });
      if (!response.ok) {
        const contentType = response.headers.get('content-type');
        if (contentType && contentType.includes('application/json')) {
          const errorData = await response.json();
          throw new Error(errorData.error || 'Failed to fetch transcribed PDF');
        }
        throw new Error('Failed to fetch transcribed PDF');
      }
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `transcribed_${result.session_id}.pdf`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      window.URL.revokeObjectURL(url);
    } catch (err) {
      setError(`Failed to download transcribed PDF: ${err.message}`);
    }
  };

  useEffect(() => {
    if (result?.pages) {
      const fetchPromises = result.pages.map((page, index) =>
        page.predicted_text_url
          ? fetch(`http://localhost:5000${page.predicted_text_url}`)
            .then(res => {
              if (!res.ok) throw new Error(`Failed to fetch text for page ${index + 1}`);
              return res.text();
            })
            .catch(err => {
              console.error(`Failed to fetch page ${index + 1} text:`, err);
              return `Error: Text not available for page ${index + 1}`;
            })
          : Promise.resolve(`Error: No text URL for page ${index + 1}`)
      );
      Promise.all(fetchPromises)
        .then(texts => setPredictedText(texts.join('\n\n--- Page Break ---\n\n')))
        .catch(err => {
          setError('Failed to fetch predicted texts for pages');
          console.error(err);
        });
    }
  }, [result]);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      const type = selectedFile.type.startsWith('image/') ? 'image' : selectedFile.type === 'application/pdf' ? 'pdf' : null;
      if (!type) {
        setError('Unsupported file type. Please upload an image or PDF.');
        return;
      }
      if (file) URL.revokeObjectURL(file);
      setFile(URL.createObjectURL(selectedFile));
      setFileType(type);
      setResult(null);
      setError(null);
      setPredictedText('');
      setOriginalText('');
      setMetrics(null);
      setShowMetricsButton(false); // Reset Metrics button visibility
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const selectedFile = e.dataTransfer.files[0];
    if (selectedFile) {
      const type = selectedFile.type.startsWith('image/') ? 'image' : selectedFile.type === 'application/pdf' ? 'pdf' : null;
      if (!type) {
        setError('Unsupported file type. Please upload an image or PDF.');
        return;
      }
      if (file) URL.revokeObjectURL(file);
      setFile(URL.createObjectURL(selectedFile));
      setFileType(type);
      setResult(null);
      setError(null);
      setPredictedText('');
      setOriginalText('');
      setMetrics(null);
      setShowMetricsButton(false); // Reset Metrics button visibility
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const handleSubmit = async () => {
    if (!file || !fileInputRef.current.files[0]) {
      setError('Please select a file before submitting');
      return;
    }
    setIsLoading(true);
    setError(null);
    const uploadedFile = fileInputRef.current.files[0];
    const formData = new FormData();
    formData.append('file', uploadedFile);
    formData.append('model', selectedModel);
    try {
      const response = await fetch('http://localhost:5000/upload', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      if (data.error) {
        setError(data.error);
      } else {
        setResult(data);
      }
    } catch (err) {
      setError('Failed to process file');
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-100 p-6">
      <h1 className="text-3xl font-bold text-center mb-6">Kannada OCR App</h1>

      <div className="flex justify-between items-center w-full max-w-2xl mx-auto">
        <div className="flex items-center">
          <label
            htmlFor="file-upload"
            className="bg-blue-500 text-white px-6 py-3 rounded hover:bg-blue-600 cursor-pointer text-lg font-semibold"
            aria-label="Choose file for upload"
          >
            Choose File
          </label>
          <input
            id="file-upload"
            type="file"
            accept="image/*,application/pdf"
            onChange={handleFileChange}
            ref={fileInputRef}
            className="hidden"
          />
        </div>
        <div className="flex items-center">
          <label htmlFor="modelSelect" className="text-gray-600 mr-2">
            Select Model:
          </label>
          <select
            id="modelSelect"
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
            className="border border-gray-300 rounded p-2"
            aria-label="Select OCR model"
          >
            <option value="densenet121_attention">DenseNet121 + Attention</option>
            <option value="botnet_resnet50">BoTNet ResNet50</option>
            <option value="vit_base">Vision Transformer (ViT)</option>
          </select>
        </div>
        <button
          onClick={handleSubmit}
          className="bg-green-500 text-white px-6 py-3 rounded hover:bg-green-600 text-lg font-semibold"
          aria-label="Submit file for OCR processing"
        >
          Submit
        </button>
      </div>

      {error && (
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-6 max-w-2xl mx-auto">
          {error}
        </div>
      )}

      {isLoading && (
        <div className="text-center mb-6">
          <p className="text-lg">Processing file...</p>
        </div>
      )}

      {result && (
        <div className="space-y-8 max-w-6xl mx-auto">
          <div className="flex flex-col lg:flex-row gap-6">
            <div className="flex-1">
              <h2 className="text-xl font-semibold mb-2">Predicted Text</h2>
              <pre className="bg-white p-4 rounded shadow whitespace-pre-wrap min-h-32 border">
                {predictedText || 'No text available'}
              </pre>
              {fileType === 'pdf' && (
                <div className="mt-4">
                  <button
                    onClick={handleDownloadTranscribedPDF}
                    className="bg-purple-500 text-white px-6 py-3 rounded hover:bg-purple-600 text-lg font-semibold"
                    aria-label="Download transcribed PDF"
                  >
                    Download Transcribed PDF
                  </button>
                </div>
              )}
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

          <div className="text-center">
            <button
              onClick={handleCalculateMetrics}
              className="bg-blue-500 text-white px-6 py-3 rounded hover:bg-blue-600 text-lg font-semibold"
              disabled={!originalText.trim() || !predictedText.trim()}
              aria-label="Calculate CER and WER"
            >
              Calculate CER and WER
            </button>
          </div>

          {showMetricsButton && (
            <div className="text-center">
              <button
                onClick={handleDownloadTranscribedPDF}
                className="bg-indigo-500 text-white px-6 py-3 rounded hover:bg-indigo-600 text-lg font-semibold"
                aria-label="Download Metrics PDF"
              >
                Metrics
              </button>
            </div>
          )}

          {metrics && (
            <div className="bg-white p-6 rounded shadow">
              <h2 className="text-xl font-semibold mb-4">CER and WER Metrics</h2>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="p-4 bg-blue-50 rounded">
                  <p className="text-lg font-semibold text-blue-800">Character Error Rate (CER)</p>
                  <p className="text-2xl font-bold text-blue-600">{metrics.CER.percentage}%</p>
                  <p className="text-sm text-gray-600">Edit distance: {metrics.CER.editDistance}/{metrics.CER.totalCharacters}</p>
                </div>
                <div className="p-4 bg-green-50 rounded">
                  <p className="text-lg font-semibold text-green-800">Word Error Rate (WER)</p>
                  <p className="text-2xl font-bold text-green-600">{metrics.WER.percentage}%</p>
                  <p className="text-sm text-gray-600">Edit distance: {metrics.WER.editDistance}/{metrics.WER.totalWords}</p>
                </div>
              </div>
            </div>
          )}

          <div className="bg-white p-6 rounded shadow">
            <h2 className="text-xl font-semibold mb-4">Original File</h2>
            {fileType === 'image' ? (
              <img
                src={file}
                alt="Uploaded"
                className="border border-gray-300 max-w-full h-auto rounded shadow"
              />
            ) : fileType === 'pdf' ? (
              <Document file={file}>
                <Page pageNumber={1} className="border border-gray-300 max-w-full h-auto rounded shadow" />
              </Document>
            ) : (
              <p className="text-gray-500">File preview not available</p>
            )}
          </div>

          <div className="bg-white p-6 rounded shadow">
            <h2 className="text-2xl font-semibold mb-6">Segmentation & Predictions</h2>
            {result.pages ? (
              result.pages.map((page, pageIndex) => (
                <div key={pageIndex} className="mb-8 border border-gray-200 rounded-lg p-4">
                  <h3 className="text-xl font-bold mb-4 text-purple-800">Page {pageIndex + 1}</h3>
                  {page.error ? (
                    <p className="text-red-600">{page.error}</p>
                  ) : (
                    Object.entries(organizeSegmentationData(page.segmentation, page.predictions)).map(([lineNum, lineInfo]) => (
                      <div key={lineNum} className="mb-8 border border-gray-200 rounded-lg p-4">
                        <h3 className="text-lg font-semibold mb-4 text-blue-800">Line {lineNum}:</h3>
                        <div className="mb-4 p-4 bg-gray-50 rounded border-2 border-dashed border-gray-300">
                          {lineInfo.lineData && lineInfo.lineData.line_url ? (
                            <img
                              src={`http://localhost:5000${lineInfo.lineData.line_url}?t=${Date.now()}`}
                              alt={`Line ${lineNum}`}
                              className="max-w-full h-auto border rounded mx-auto"
                              onError={(e) => {
                                e.target.src = '/fallback-image.png';
                                console.error(`Line image load failed: ${e.target.src}`);
                              }}
                            />
                          ) : (
                            <p className="text-gray-500 text-center">Line Image (requires line_url from backend)</p>
                          )}
                        </div>
                        <div className="space-y-4">
                          {Object.entries(lineInfo.words).map(([wordNum, wordInfo]) => (
                            <div key={wordNum} className="ml-4 border-l-2 border-gray-300 pl-4">
                              <h4 className="text-md font-medium mb-2 text-green-700">Word {wordNum}:</h4>
                              <div className="mb-3 p-3 bg-gray-50 rounded border border-gray-200">
                                {wordInfo.wordData && wordInfo.wordData.word_url ? (
                                  <img
                                    src={`http://localhost:5000${wordInfo.wordData.word_url}?t=${Date.now()}`}
                                    alt={`Word ${wordNum} in Line ${lineNum}`}
                                    className="max-w-full h-auto border rounded mx-auto"
                                    onError={(e) => {
                                      e.target.src = '/fallback-image.png';
                                      console.error(`Word image load failed: ${e.target.src}`);
                                    }}
                                  />
                                ) : (
                                  <p className="text-gray-500 text-center text-sm">Word Image (requires word_url from backend)</p>
                                )}
                              </div>
                              <div className="flex flex-wrap gap-3 ml-4">
                                {wordInfo.characters.map((char, charIndex) => (
                                  <div key={charIndex} className="text-center">
                                    <div className="w-16 h-16 border border-gray-300 rounded bg-white flex items-center justify-center mb-1">
                                      <img
                                        src={`http://localhost:5000${char.char_url}?t=${Date.now()}`}
                                        alt={`Character ${char.char}`}
                                        className="max-w-full max-h-full object-contain"
                                        onError={(e) => {
                                          e.target.src = '/fallback-image.png';
                                          console.error(`Character image load failed: ${e.target.src}`);
                                        }}
                                      />
                                    </div>
                                    <div className="text-sm font-mono bg-gray-100 px-1 py-0.5 rounded min-w-[2rem]">
                                      {char.label || char.combined_char || '?'}
                                    </div>
                                    <div className="text-xs text-gray-500 mt-1">
                                      {char.type}
                                    </div>
                                  </div>
                                ))}
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    ))
                  )}
                </div>
              ))
            ) : (
              <p className="text-gray-500">No segmentation data available</p>
            )}
          </div>

          <div className="bg-white p-6 rounded shadow">
            <h2 className="text-xl font-semibold mb-4">Processing Metrics</h2>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="text-center p-3 bg-blue-50 rounded">
                <p className="text-sm text-gray-600">Total Lines</p>
                <p className="text-xl font-bold text-blue-600">{result.metrics.total_lines}</p>
              </div>
              <div className="text-center p-3 bg-green-50 rounded">
                <p className="text-sm text-gray-600">Total Words</p>
                <p className="text-xl font-bold text-green-600">{result.metrics.total_words}</p>
              </div>
              <div className="text-center p-3 bg-purple-50 rounded">
                <p className="text-sm text-gray-600">Total Characters</p>
                <p className="text-xl font-bold text-purple-600">{result.metrics.total_chars}</p>
              </div>
              <div className="text-center p-3 bg-orange-50 rounded">
                <p className="text-sm text-gray-600">Avg Confidence</p>
                <p className="text-xl font-bold text-orange-600">{result.metrics.avg_confidence.toFixed(2)}</p>
              </div>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-4">
              <div className="text-center p-3 bg-gray-50 rounded">
                <p className="text-sm text-gray-600">Total Ottaksharas</p>
                <p className="text-lg font-semibold">{result.metrics.total_ottaksharas}</p>
              </div>
              <div className="text-center p-3 bg-gray-50 rounded">
                <p className="text-sm text-gray-600">Avg Character Area</p>
                <p className="text-lg font-semibold">{result.metrics.avg_char_area.toFixed(2)}</p>
              </div>
              <div className="text-center p-3 bg-gray-50 rounded">
                <p className="text-sm text-gray-600">Total Predictions</p>
                <p className="text-lg font-semibold">{result.metrics.total_predictions}</p>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;