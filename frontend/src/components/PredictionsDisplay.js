import React from 'react';
import { Card, CardContent, Typography } from '@mui/material';

const PredictionsDisplay = ({ predictions }) => {
  if (!predictions || predictions.length === 0) return null;

  return (
    <div>
      {predictions.map((prediction, index) => (
        <Card key={index} style={{ margin: '20px 0' }}>
          <CardContent>
            <Typography variant="h6">
              Product: {prediction.productName}
            </Typography>
            <Typography variant="body2">
              Confidence: {Math.round(prediction.confidence * 100)}%
            </Typography>
          </CardContent>
        </Card>
      ))}
    </div>
  );
};

export default PredictionsDisplay;
