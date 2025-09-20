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