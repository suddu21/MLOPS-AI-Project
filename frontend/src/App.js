import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [transaction, setTransaction] = useState({
    Time: 0, V1: 0, V2: 0, V3: 0, V4: 0, V5: 0, V6: 0, V7: 0, V8: 0, V9: 0,
    V10: 0, V11: 0, V12: 0, V13: 0, V14: 0, V15: 0, V16: 0, V17: 0, V18: 0,
    V19: 0, V20: 0, V21: 0, V22: 0, V23: 0, V24: 0, V25: 0, V26: 0, V27: 0,
    V28: 0, Amount: 0
  });
  const [result, setResult] = useState(null);

  const handleChange = (e) => {
    setTransaction({ ...transaction, [e.target.name]: parseFloat(e.target.value) });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const response = await axios.post('http://localhost:8000/predict', transaction);
      setResult(response.data);
    } catch (error) {
      setResult({ error: 'Prediction failed' });
    }
  };

  return (
    <div className="container mx-auto p-4">
      <h1 className="text-2xl font-bold mb-4">Fraud Detection</h1>
      <form onSubmit={handleSubmit} className="space-y-4">
        {Object.keys(transaction).map((key) => (
          <div key={key}>
            <label className="block">{key}</label>
            <input
              type="number"
              name={key}
              value={transaction[key]}
              onChange={handleChange}
              className="border p-2 w-full"
            />
          </div>
        ))}
        <button type="submit" className="bg-blue-500 text-white p-2 rounded">
          Predict
        </button>
      </form>
      {result && (
        <div className="mt-4">
          <h2 className="text-xl">Result</h2>
          {result.error ? (
            <p className="text-red-500">{result.error}</p>
          ) : (
            <p>
              Prediction: {result.prediction ? 'Fraud' : 'Legitimate'} (Probability: {result.probability.toFixed(2)})
            </p>
          )}
        </div>
      )}
    </div>
  );
}

export default App;