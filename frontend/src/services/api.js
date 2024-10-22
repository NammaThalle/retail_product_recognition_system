import axios from 'axios';

// const API_URL = 'https://your-api-gateway-url.amazonaws.com/recognize';  // Replace with your API Gateway URL
const API_URL = 'http://localhost:5000/recognize';  // Local Flask server URL

export const uploadImage = (file) => {
  const formData = new FormData();
  formData.append('file', file);

  return axios.post(API_URL, formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
};