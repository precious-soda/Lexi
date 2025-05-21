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
