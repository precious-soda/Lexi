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
// GitHub: https://github.com/precious-soda

function ResultsDisplay({ results }) {
  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-semibold mb-2">Predicted Text</h2>
        <pre className="bg-gray-100 p-4 rounded whitespace-pre-wrap">
          {results.predicted_text}
        </pre>
      </div>
      <div>
        <h2 className="text-2xl font-semibold mb-2">Visualization</h2>
        <img
          src={results.visualization}
          alt="Visualization"
          className="w-full rounded"
        />
      </div>
      <div>
        <h2 className="text-2xl font-semibold mb-2">Metrics</h2>
        <ul className="list-disc pl-5">
          <li>Total Lines: {results.metrics.total_lines}</li>
          <li>Total Words: {results.metrics.total_words}</li>
          <li>Total Characters: {results.metrics.total_chars}</li>
          <li>Average Character Area: {results.metrics.avg_area}</li>
        </ul>
      </div>
      <div>
        <h2 className="text-2xl font-semibold mb-2">Downloads</h2>
        <div className="space-x-4">
          <a
            href={results.text_file}
            download
            className="text-blue-500 hover:underline"
          >
            Download Predicted Text
          </a>
          <a
            href={`/api/download_dir/${results.output_dir.split('/output/')[1]}`}
            download
            className="text-blue-500 hover:underline"
          >
            Download All Files (ZIP)
          </a>
        </div>
      </div>
    </div>
  );
}

export default ResultsDisplay;
