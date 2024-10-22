import React, { useState } from 'react';
import { Button, TextField, CircularProgress } from '@mui/material';

const UploadImage = ({ onImageSubmit }) => {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleSubmit = async () => {
    if (!file) return alert('Please upload an image');
    setLoading(true);
    await onImageSubmit(file);
    setLoading(false);
  };

  return (
    <div>
      <input
        accept="image/*"
        type="file"
        onChange={handleFileChange}
        style={{ display: 'none' }}
        id="upload-button"
      />
      <label htmlFor="upload-button">
        <Button variant="contained" component="span">
          Upload Image
        </Button>
      </label>
      <Button
        variant="contained"
        color="primary"
        onClick={handleSubmit}
        disabled={loading}
      >
        Submit
      </Button>
      {loading && <CircularProgress />}
    </div>
  );
};

export default UploadImage;
