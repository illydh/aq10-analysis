import React, { useState } from 'react';
import PatientForm from './components/PatientForm';

export default function App() {
  const [result, setResult] = useState(null);

  return (
    <div style={{ padding: 20 }}>
      <h1>ASD Clinical Support</h1>
      <PatientForm onResult={setResult} />
      {result && (
        <div style={{ marginTop: 20 }}>
          <h2>Result</h2>
          <p><strong>Probability:</strong> {(result.probability*100).toFixed(1)}%</p>
          <p><strong>Decision:</strong> {result.decision ? 'Positive' : 'Negative'}</p>
          <h3>Summary & Next Steps</h3>
          <p>{result.summary}</p>
        </div>
      )}
    </div>
  );
}