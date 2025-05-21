import { useState, useRef } from 'react';

function ImageUploader({ onUpload }) {
  const [preview, setPreview] = useState(null);
  const fileInputRef = useRef(null);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setPreview(URL.createObjectURL(file));
    }
  };

  const handleSubmit = () => {
    const file = fileInputRef.current.files[0];
    if (file) {
      onUpload(file);
    } else {
      alert('Please select an image');
    }
  };

  return (
    <div className="mb-6 text-center">
      <input
        type="file"
        accept="image/*"
        ref={fileInputRef}
        onChange={handleFileChange}
        className="mb-4"
      />
      {preview && (
        <img
          src={preview}
          alt="Preview"
          className="mx-auto mb-4 max-w-xs rounded"
        />
      )}
      <button
        onClick={handleSubmit}
        className="bg-green-500 text-white px-4 py-2 rounded hover:bg-green-600"
      >
        Process Image
      </button>
    </div>
  );
}

export default ImageUploader;