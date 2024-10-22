import React, { useState } from 'react';
import { AppBar, Toolbar, Typography, Container, Button, Grid, Card, CardContent, CircularProgress, Switch } from '@mui/material';
import { Box } from '@mui/system';
import { createTheme, ThemeProvider } from '@mui/material/styles';
import UploadIcon from '@mui/icons-material/Upload';
import PhotoCamera from '@mui/icons-material/PhotoCamera';
import Brightness4Icon from '@mui/icons-material/Brightness4';
import Brightness7Icon from '@mui/icons-material/Brightness7';
import axios from 'axios'; // Import axios to make API requests

// Define light and dark themes with background color settings
const lightTheme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#ff4081',
    },
    background: {
      default: '#f5f5f5',  // Light background color
      paper: '#fff',  // Card background
    },
  },
});

const darkTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#90caf9',
    },
    secondary: {
      main: '#f48fb1',
    },
    background: {
      default: '#121212',  // Dark background color
      paper: '#1e1e1e',  // Card background
    },
  },
});

function App() {
  const [darkMode, setDarkMode] = useState(false);  // Night mode toggle state
  const [loading, setLoading] = useState(false);
  const [predictions, setPredictions] = useState([]);

  const handleUpload = async (event) => {
    setLoading(true);
    const file = event.target.files[0];
    const formData = new FormData();
    formData.append('file', file); // Prepare the image file for the backend

    try {
      // Send image to backend
      const response = await axios.post('http://localhost:8000/predict', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      // Update the predictions based on backend response
      setPredictions(response.data.predictions); 
      setLoading(false);
    } catch (error) {
      console.error("Error uploading file", error);
      setLoading(false);
    }
  };

  return (
    <ThemeProvider theme={darkMode ? darkTheme : lightTheme}>
      <Box
        sx={{
          bgcolor: 'background.default',  // Apply background color based on the theme
          color: 'text.primary',
          minHeight: '100vh',  // Ensures the background covers the entire page
        }}
      >
        {/* AppBar with night mode toggle */}
        <AppBar position="static" color="primary">
          <Toolbar>
            <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
              Retail Product Recognition
            </Typography>
            {/* Night mode toggle switch */}
            <Switch
              checked={darkMode}
              onChange={() => setDarkMode(!darkMode)}
              icon={<Brightness7Icon />}
              checkedIcon={<Brightness4Icon />}
              color="default"
            />
          </Toolbar>
        </AppBar>

        {/* Main Content */}
        <Container sx={{ mt: 4 }}>
          <Grid container spacing={4} justifyContent="center">
            <Grid item xs={12} md={6}>
              {/* Image Upload Section */}
              <Card sx={{ textAlign: 'center', padding: 4 }}>
                <Typography variant="h5" gutterBottom>
                  Upload a Product Image
                </Typography>
                <Button
                  variant="contained"
                  component="label"
                  startIcon={<PhotoCamera />}
                  sx={{ marginTop: 2, marginBottom: 2 }}
                >
                  Upload Image
                  <input hidden accept="image/*" type="file" onChange={handleUpload} />
                </Button>

                {/* Loading Indicator */}
                {loading && <CircularProgress color="secondary" />}
              </Card>
            </Grid>
          </Grid>

          {/* Display Predictions */}
          <Box sx={{ mt: 6 }}>
            <Typography variant="h5" align="center" gutterBottom>
              Recognition Results
            </Typography>

            <Grid container spacing={3} justifyContent="center">
              {predictions.map((prediction, index) => (
                <Grid item xs={12} sm={6} md={4} key={index}>
                  <Card sx={{ textAlign: 'center', padding: 2 }}>
                    <CardContent>
                      <Typography variant="h6">{prediction.name}</Typography>
                      <Typography variant="body1" color="text.secondary">
                        Confidence: {prediction.confidence}%
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
              ))}
            </Grid>
          </Box>
        </Container>
      </Box>
    </ThemeProvider>
  );
}

export default App;
