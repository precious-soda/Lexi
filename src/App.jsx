import { useState, useRef } from 'react';

function App() {
  const [image, setImage] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const fileInputRef = useRef(null);
  const canvasRef = useRef(null);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setImage(URL.createObjectURL(file));
      setResult(null);
      setError(null);
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
      if (data.error) {
        setError(data.error);
      } else {
        setResult(data);
        drawBoundingBoxes(data);
      }
    } catch (err) {
      setError('Failed to process image');
    } finally {
      setIsLoading(false);
    }
  };

  const drawBoundingBoxes = (data) => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const img = new Image();
    img.src = image;
    img.onload = () => {
      canvas.width = img.width;
      canvas.height = img.height;
      ctx.drawImage(img, 0, 0);
      // Draw line boxes
      Object.values(data.segmentation.lines).forEach((line) => {
        const [x, y, w, h] = line.bbox;
        ctx.strokeStyle = 'blue';
        ctx.lineWidth = 2;
        ctx.strokeRect(x, y, w, h);
        // Draw word boxes
        Object.values(line.words).forEach((word) => {
          const [wx, wy, ww, wh] = word.bbox;
          ctx.strokeStyle = 'green';
          ctx.lineWidth = 1;
          ctx.strokeRect(x + wx, y + wy, ww, wh);
          // Draw character boxes
          word.characters.forEach((char) => {
            const [cx, cy, cw, ch] = char.bbox;
            ctx.strokeStyle = char.type === 'main' ? 'red' : 'purple';
            ctx.lineWidth = 1;
            ctx.strokeRect(x + wx + cx, y + wy + cy, cw, ch);
          });
        });
      });
    };
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
        <button
          onClick={() => fileInputRef.current.click()}
          className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600"
        >
          Upload Image
        </button>
        <p className="mt-2 text-gray-600">or drag and drop an image here</p>
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
          {/* Image with Bounding Boxes */}
          <div>
            <h2 className="text-xl font-semibold mb-2">Original Image</h2>
            <img
              src={image}
              alt="Uploaded"
              className="border border-gray-300 max-w-full h-auto"
            />
          </div>

          {/* Predictions Table */}
          <div>
            <h2 className="text-xl font-semibold mb-2">Predicted Characters</h2>
            <table className="w-full border-collapse border border-gray-300">
              <thead>
                <tr className="bg-gray-200">
                  <th className="border border-gray-300 p-2">Line</th>
                  <th className="border border-gray-300 p-2">Word</th>
                  <th className="border border-gray-300 p-2">Char</th>
                  <th className="border border-gray-300 p-2">Type</th>
                  <th className="border border-gray-300 p-2">Label</th>
                  {/* <th className="border border-gray-300 p-2">Confidence</th> */}
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
                    {/* <td className="border border-gray-300 p-2">{pred.confidence.toFixed(2)}</td> */}
                    <td className="border border-gray-300 p-2">
                      <img src={pred.char_url} alt="Character" className="w-16 h-16 object-contain" />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Metrics */}
          {/* <div>
            <h2 className="text-xl font-semibold mb-2">Metrics</h2>
            <div className="bg-white p-4 rounded shadow">
              <p><strong>Total Lines:</strong> {result.metrics.total_lines}</p>
              <p><strong>Total Words:</strong> {result.metrics.total_words}</p>
              <p><strong>Total Characters:</strong> {result.metrics.total_chars}</p>
              <p><strong>Main Characters:</strong> {result.metrics.total_main_chars}</p>
              <p><strong>Ottaksharas:</strong> {result.metrics.total_ottaksharas}</p>
              <p><strong>Average Character Area:</strong> {result.metrics.avg_char_area.toFixed(2)}</p>
              <p><strong>Total Predictions:</strong> {result.metrics.total_predictions}</p>
              <p><strong>Average Confidence:</strong> {result.metrics.avg_confidence.toFixed(2)}</p>
            </div>
          </div> */}
        </div>
      )}
    </div>
  );
}

export default App;